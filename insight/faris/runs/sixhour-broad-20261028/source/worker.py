"""Fit one predeclared design; query labels never enter a model or stopping rule.

Fitted model/preprocessor state stays in RAM. The only per-case saved artifact is
the private compact prediction file, accompanied by source/config/row identities.
"""
import os
for _name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_name] = '2'
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import argparse
import copy
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import resource
import socket
import sys
import time
import traceback

import numpy as np
import pandas as pd

from common import R, load, dump, sha, ids_sha, utc, check_seal, tasks, catalog


def model_deadline(clock):
    """Accept the coordinator's explicit model cutoff, otherwise total ceiling."""
    return float(clock.get('model_deadline_epoch', clock.get('model_end_epoch', clock['deadline_epoch'])))


def check_time(clock, admission):
    end = model_deadline(clock)
    if admission:
        end = min(end, float(clock['admission_end_epoch']))
    if time.time() >= end:
        raise TimeoutError('Original campaign admission/model cutoff reached; no clock extension.')


def versions():
    result = {'python': sys.version}
    for name in ['numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'lightgbm',
                 'catboost', 'xgboost', 'torch', 'tabm', 'rtdl-num-embeddings',
                 'structured-data-models']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not installed; family admission must verify required dependencies'
    return result


def resolved_spec(design, fold, clock):
    """Preserve nested estimator parameters and top-level classical recipes."""
    spec = copy.deepcopy(design)
    parameters = copy.deepcopy(design.get('parameters', {}))
    spec.update(parameters)
    spec['parameters'] = parameters
    spec['inner_split_seed'] = 20261029 + int(fold)
    # The time limit is execution control, not a silently shortened epoch policy.
    spec['max_seconds'] = min(float(spec.get('max_seconds', 900)),
                              max(1., model_deadline(clock) - time.time() - 15.))
    if design['family'] in {'tabm', 'resnet', 'mlp'}:
        spec.setdefault('kind', design['family'])
    return spec


def expected_checkpoint(size):
    values = load('CHECKPOINTS.json')
    entry = values[size]
    expected = entry if isinstance(entry, str) else entry['sha256']
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Invalid immutable pretrained checkpoint hash for ' + size)
    physical = load('ENVIRONMENT.json')['checkpoint_' + size]
    path = Path(physical).resolve()
    assert path.is_relative_to(Path.home().resolve()), 'Checkpoint must stay in the actual user home'
    assert sha(path) == expected, 'Physical checkpoint differs from immutable identity'
    return str(path), expected


def fit(frame, ctx, y, query, design, spec, seed, device):
    family = design['family']
    if family.startswith('kumo'):
        if device != 'cuda':
            raise RuntimeError('Kumo requires an admitted CUDA backend; no CPU fallback.')
        assert len(frame.columns) <= 99
        size = spec.get('size', 'medium' if family == 'kumo_medium' else 'large')
        checkpoint, digest = expected_checkpoint(size)
        sys.path.insert(0, str(R / 'source/vendor'))
        from kumo_engine import predict
        estimators = int(spec.get('estimators', spec.get('num_estimators', 8)))
        assert estimators in {2, 8, 16}
        prediction, audit = predict(
            frame.iloc[ctx], frame.iloc[query], list(frame.columns), y, checkpoint,
            seed, engine=spec.get('engine', 'concat_context_vocab'),
            estimators=estimators, pred_batch=int(spec.get('prediction_batch', 256)),
            mem_fraction=float(spec.get('memory_fraction', .7)),
            size=size, expected_sha=digest,
        )
        audit.update(checkpoint_sha256=digest, size=size,
                     vendor_inference_source_sha256=sha(R / 'source/vendor/kumo_engine.py'),
                     correction_used_for_early_stopping=False,
                     fitted_weights_saved=False, KV_caches_saved=False,
                     execution_backend='CUDA AMP float16')
        return prediction, audit
    if family in {'neural', 'tabm', 'resnet', 'mlp'}:
        from models_neural import train_predict
        return train_predict(frame, ctx, y, query, spec, seed, device=device)
    from models_classical import train_predict
    return train_predict(frame, ctx, y, query, spec, seed, device=device)


def run(task_id, attempt=1, pilot=False):
    assert re.fullmatch('[A-Za-z0-9_.-]+', task_id) and '..' not in task_id
    assert isinstance(attempt, int) and attempt > 0
    started = time.time()
    folder = R / f'runs/{task_id}/attempt{attempt}'
    folder.mkdir(parents=True, exist_ok=False)
    try:
        clock = check_seal()
        check_time(clock, admission=True)
        config = load('CONFIG.json')
        task = tasks()[task_id]
        design_id = task['design']
        design = catalog()[design_id]
        gate = None
        if pilot:
            assert task_id in config['pilot_tasks'], 'Unregistered pilot task'
        else:
            gate = load('GATE.json')
            assert gate['status'] == 'PASS', 'Scientific admission gate did not pass'
            if 'contract_sha256' in gate:
                assert gate['contract_sha256'] == sha(R / 'VALIDATION_CONTRACT.json')
            if 'config_sha256' in gate:
                assert gate['config_sha256'] == sha(R / 'CONFIG.json')
            if 'source_seal_sha256' in gate:
                assert gate['source_seal_sha256'] == sha(R / 'SOURCE_SEAL.json')
            qualification = load('GATE_FAMILY_' + design['family'] + '.json')
            assert qualification['status'] == 'PASS', 'Family runtime qualification required'
            for key, name in [('clock_sha256','CLOCK.json'), ('contract_sha256','VALIDATION_CONTRACT.json'), ('source_seal_sha256','SOURCE_SEAL.json'), ('config_sha256','CONFIG.json')]:
                assert qualification[key] == sha(R / name)
        fold = int(task['fold'])
        assert fold in range(5)
        packet = R / f'packets/fold{fold}'
        manifest = json.loads((packet / 'MANIFEST.json').read_text())
        manifest_sha = sha(packet / 'MANIFEST.json')
        if gate is not None and 'packet_manifests' in gate:
            assert manifest_sha == gate['packet_manifests'][str(fold)]
        assert sha(packet / 'roles.npz') == manifest['roles_sha256']
        with np.load(packet / 'roles.npz') as roles:
            assert not {'y_outer', 'y_validation', 'target', 'labels'} & set(roles.files)
            ctx = roles['context'].astype(np.int64)
            correction = roles['correction'].astype(np.int64)
            validation = roles['validation'].astype(np.int64)
            # Only the permitted context labels are materialized by the worker.
            y = roles['y_context'].astype(np.int8)
        query = np.r_[correction, validation]
        assert (len(ctx), len(correction), len(validation)) == (50000, 6000, 14000)
        assert len(np.unique(np.r_[ctx, correction, validation])) == 70000
        assert len(y) == len(ctx) and set(np.unique(y)) == {0, 1}
        columns = manifest['views'][design['view']]
        assert len(columns) > 0 and len(set(columns)) == len(columns)
        assert not {'target', 'application_id', 'customer_id', 'split', 'y_outer', 'y_validation'} & set(columns)
        if design['family'].startswith('kumo'):
            assert len(columns) <= 99
        if set(columns).issubset(set(manifest['feature_columns'])):
            feature_filename = 'features.parquet'
            feature_digest = manifest['feature_sha256']
        else:
            assert design['resource'] == 'cpu', 'Full bank is admitted only on the lab CPU backends'
            feature_filename = 'all_features.parquet'
            feature_digest = manifest['full_feature_sha256']
        assert sha(packet / feature_filename) == feature_digest
        frame = pd.read_parquet(packet / feature_filename, columns=columns)
        assert len(frame) == 70000 and list(frame.columns) == columns
        spec = resolved_spec(design, fold, clock)
        spec['inner_split_seed'] = int(manifest['inner_split_seed'])
        assert spec['inner_split_seed'] == 20261029 + fold
        spec['early_stop_fraction'] = .15
        from sklearn.model_selection import train_test_split
        inner_train, inner_stop = train_test_split(
            np.arange(len(ctx)), test_size=.15, stratify=y,
            random_state=spec['inner_split_seed'],
        )
        assert (len(inner_train), len(inner_stop)) == (42500, 7500)
        expected_inner_train_sha = ids_sha(ctx[inner_train])
        expected_inner_stop_sha = ids_sha(ctx[inner_stop])
        seed = int(task['seed']) + fold
        device = 'cpu' if design['resource'] == 'cpu' else 'cuda'
        if device == 'cuda':
            import torch
            assert torch.cuda.is_available(), 'Requested GPU is unavailable; no silent CPU switch'
            torch.cuda.reset_peak_memory_stats()
        record = {
            'status': 'running', 'task_id': task_id, 'task': task,
            'design_id': design_id, 'design': design, 'resolved_spec': spec,
            'attempt': attempt, 'pilot': bool(pilot), 'fold': fold, 'seed': seed,
            'view': design['view'], 'columns': columns, 'predictor_count': len(columns),
            'context_rows': len(ctx), 'correction_rows': len(correction),
            'validation_rows': len(validation), 'query_rows': len(query),
            'context_rows_sha256': ids_sha(ctx), 'prediction_rows_sha256': ids_sha(query),
            'context_labels_sha256': ids_sha(y),
            'inner_split_seed': spec['inner_split_seed'],
            'declared_inner_train_rows_sha256': expected_inner_train_sha,
            'declared_inner_stop_rows_sha256': expected_inner_stop_sha,
            'packet_manifest_sha256': manifest_sha,
            'feature_table_filename': feature_filename,
            'feature_sha256': feature_digest, 'roles_sha256': manifest['roles_sha256'],
            'source_sha256': sha(__file__), 'source_seal_sha256': sha(R / 'SOURCE_SEAL.json'),
            'config_sha256': sha(R / 'CONFIG.json'), 'contract_sha256': sha(R / 'VALIDATION_CONTRACT.json'),
            'clock_sha256': sha(R / 'CLOCK.json'),
            'host': socket.gethostname(), 'device': device,
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'environment': versions(), 'UTC': utc(),
            'outer_labels_loaded': False, 'correction_labels_loaded': False,
            'correction_labels_used_for_training': False,
            'query_labels_used_for_early_stopping': False,
            'pilot_parameters_changed': False,
        }
        if (R / 'EXTRA_DESIGNS.jsonl').exists():
            record['append_only_extra_design_ledger_sha256_at_start'] = sha(R / 'EXTRA_DESIGNS.jsonl')
        dump(record, folder / 'START.json')
        check_time(clock, admission=True)
        prediction, audit = fit(frame, ctx, y, query, design, spec, seed, device)
        if 'inner_train_rows_sha256' in audit:
            assert audit['inner_train_rows_sha256'] == expected_inner_train_sha
        if 'stopping_rows_sha256' in audit:
            assert audit['stopping_rows_sha256'] == expected_inner_stop_sha
        prediction = np.asarray(prediction, dtype=np.float64)
        assert prediction.shape == (20000,) and np.isfinite(prediction).all()
        assert np.all((prediction >= 0) & (prediction <= 1))
        np.savez_compressed(folder / 'prediction.npz', rows=query, raw=prediction)
        # Preserve a valid partial output even if execution finishes too late.
        check_time(clock, admission=False)
        peak_torch = 0
        if device == 'cuda':
            import torch
            peak_torch = int(torch.cuda.max_memory_allocated())
        result = {
            'status': 'complete', 'task_id': task_id, 'attempt': attempt, 'pilot': bool(pilot),
            'seconds': time.time() - started, 'UTC': utc(), 'audit': audit,
            'prediction_sha256': sha(folder / 'prediction.npz'),
            'start_sha256': sha(folder / 'START.json'),
            'clock_sha256': sha(R / 'CLOCK.json'),
            'source_seal_sha256': sha(R / 'SOURCE_SEAL.json'),
            'max_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'peak_torch_CUDA_bytes': peak_torch,
            'GPU_memory_measurement_scope': 'PyTorch allocator only; excludes third-party CUDA allocators and parallel processes.',
            'outer_labels_loaded': False, 'correction_labels_used_for_training': False,
            'weights_saved': False, 'fitted_weights_saved': False, 'KV_cache_saved': False,
        }
        dump(result, folder / 'RESULT.json')
        print(json.dumps({'status': 'complete', 'task_id': task_id, 'seconds': result['seconds']}), flush=True)
        del frame, prediction
        gc.collect()
        return result
    except BaseException:
        dump({
            'status': 'failed', 'task_id': task_id, 'attempt': attempt,
            'pilot': bool(pilot), 'UTC': utc(), 'seconds': time.time() - started,
            'error': traceback.format_exc(),
            'prediction_file_preserved_if_present': (folder / 'prediction.npz').exists(),
            'scientific_conclusion': 'None from execution failure; preserve evidence and inspect the cause.',
            'weights_saved': False, 'KV_cache_saved': False,
        }, folder / 'FAILURE.json')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    run(args.task, args.attempt, args.pilot)
