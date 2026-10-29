"""Matched case-control versus natural-prevalence Kumo contexts.

Only the supplied 50,000 permitted context labels determine the sample. Every
positive is retained in a case-control arm and negatives are drawn without
replacement. Its natural-prevalence control has exactly the same row count.
This does not implement local case-control or claim sampling invariance of Kumo.
No analytic prior correction is applied; the registered downstream held-meta
calibration is unchanged. Case-control raw scores are not population risk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np


CHECKPOINT_SHA256 = {
    'large': 'cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613',
    'medium': '70d6aee8fd13416300afead71a5145a40edb790952e2417d87f27b3f7e57793d',
}
ENGINE_SHA256 = 'f05ccade0bfead7dae94d80965f6d99b15a7a7c0fe5abe6173d9f1499aa74697'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _array_sha(values, dtype):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def sample_context(y, negative_per_positive, sampling, seed):
    """Choose context offsets from labels alone, with deterministic sorted output.

    N = (1 + negative_per_positive) * number_of_available_positives. The design
    fails explicitly if that many unique negatives do not exist; it never clamps,
    duplicates examples, switches ratios or changes the recipe to fit resources.
    Both arms use the same registered draw-seed formula and exact N.
    """
    y = np.asarray(y, dtype=np.int8)
    assert y.ndim == 1 and len(y) == 50000
    assert set(np.unique(y)) == {0, 1}
    ratio = int(negative_per_positive)
    assert ratio in {1, 3, 5} and ratio == negative_per_positive
    assert sampling in {'case_control', 'natural_matched_count'}
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    requested_negatives = ratio * len(positive)
    assert requested_negatives <= len(negative), 'Registered ratio is infeasible; no clamping or replacement.'
    count = len(positive) + requested_negatives
    assert 1 < count < len(y), 'Natural control must be a proper subset.'
    draw_seed = (int(seed) + 73939133 + ratio * 104729) % (2**32)
    if sampling == 'case_control':
        rng = np.random.default_rng(draw_seed)
        sampled_negative = rng.choice(negative, size=requested_negatives, replace=False)
        selected = np.sort(np.r_[positive, sampled_negative].astype(np.int64))
        assert np.array_equal(selected[y[selected] == 1], positive)
        assert np.bincount(y[selected], minlength=2).tolist() == [requested_negatives, len(positive)]
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=count, random_state=draw_seed)
        selected, _ = next(splitter.split(np.zeros((len(y), 1), dtype=np.int8), y))
        selected = np.sort(np.asarray(selected, dtype=np.int64))
        assert abs(float(y[selected].mean()) - float(y.mean())) <= 1. / count
    assert len(selected) == count and len(np.unique(selected)) == count
    assert selected.min() >= 0 and selected.max() < len(y)
    return selected, draw_seed


def _checkpoint(spec, size):
    expected = CHECKPOINT_SHA256[size]
    assert spec.get('checkpoint_sha256', expected) == expected
    physical = spec.get('checkpoint_path', spec.get('checkpoint_' + size))
    if not physical:
        environment = Path(__file__).resolve().parents[1] / 'ENVIRONMENT.json'
        physical = json.loads(environment.read_text())['checkpoint_' + size]
    physical = Path(physical).resolve()
    assert physical.is_relative_to(Path.home().resolve()), 'Checkpoint must stay in the actual user home.'
    assert physical.is_file() and _sha(physical) == expected
    return str(physical), expected


def train_predict(frame, ctx, y, query, spec, seed, device='cuda'):
    """Drop-in interface for the sealed packet worker; labels enter through y only."""
    started = time.time()
    assert device == 'cuda', 'CPU Kumo is prohibited; no fallback.'
    import torch
    assert torch.cuda.is_available()
    ctx = np.asarray(ctx, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    y = np.asarray(y, dtype=np.int8)
    assert (len(frame), len(ctx), len(y), len(query)) == (70000, 50000, 50000, 20000)
    assert len(np.unique(ctx)) == len(ctx) and len(np.unique(query)) == len(query)
    assert not np.intersect1d(ctx, query).size
    assert min(ctx.min(), query.min()) >= 0 and max(ctx.max(), query.max()) < len(frame)
    columns = list(frame.columns)
    assert 1 <= len(columns) <= 99 and len(set(columns)) == len(columns)
    assert not {'target', 'application_id', 'customer_id', 'split', 'y_outer', 'y_validation'} & set(columns)
    size = spec['size']
    ratio = spec['negative_per_positive']
    sampling = spec['context_sampling']
    assert size in {'medium', 'large'}
    assert int(spec['estimators']) in {2,8}
    assert spec.get('engine', 'concat_context_vocab') == 'concat_context_vocab'
    assert int(spec.get('prediction_batch', 256)) == 256
    assert float(spec.get('memory_fraction', .7)) == .7
    assert spec.get('prior_correction', 'none') == 'none'
    assert spec.get('aggregation', 'single_context') == 'single_context'
    deadline = spec.get('model_deadline_epoch', spec.get('deadline_epoch'))
    if deadline is not None and time.time() >= float(deadline):
        raise TimeoutError('Original model deadline reached before context sampling.')
    selected, draw_seed = sample_context(y, ratio, sampling, seed)
    physical_context = ctx[selected]
    checkpoint, digest = _checkpoint(spec, size)
    engine = Path(__file__).with_name('gpu_kumo_engine.py')
    assert _sha(engine) == ENGINE_SHA256, 'Pinned inference engine changed.'
    from gpu_kumo_engine import predict
    print(json.dumps({'stage': 'case_control_context_start', 'size': size,
                      'sampling': sampling, 'negative_per_positive': ratio,
                      'context_rows': len(selected), 'class_counts': np.bincount(y[selected], minlength=2).tolist(),
                      'context_draw_seed': draw_seed, 'model_seed': int(seed)}), flush=True)
    prediction, engine_audit = predict(
        frame.iloc[physical_context], frame.iloc[query], columns, y[selected],
        checkpoint, int(seed), engine='concat_context_vocab', estimators=int(spec['estimators']),
        pred_batch=256, mem_fraction=.7, size=size, expected_sha=digest,
    )
    prediction = np.asarray(prediction, dtype=np.float64)
    assert prediction.shape == (20000,) and np.isfinite(prediction).all()
    assert np.all((prediction >= 0) & (prediction <= 1))
    assert engine_audit['first_batch_replay_error'] < 1e-7
    return prediction, {
        'family': 'kumo_casecontrol', 'size': size, 'context_sampling': sampling,
        'negative_per_positive': int(ratio), 'available_context_rows': len(ctx),
        'available_context_class_counts': np.bincount(y, minlength=2).tolist(),
        'available_context_positive_fraction': float(y.mean()),
        'sampled_context_rows': len(selected), 'sampled_context_class_counts': np.bincount(y[selected], minlength=2).tolist(),
        'sampled_context_positive_fraction': float(y[selected].mean()),
        'retained_positive_fraction': float(np.sum(y[selected]) / np.sum(y)),
        'context_draw_seed': draw_seed, 'model_seed': int(seed),
        'context_rows_sha256': _array_sha(physical_context, '<i8'),
        'context_offsets_sha256': _array_sha(selected, '<i8'),
        'context_labels_sha256': _array_sha(y[selected], 'i1'),
        'prediction_array_sha256': _array_sha(prediction, '<f8'),
        'sampling_with_replacement': False, 'sampled_offsets_sorted': True,
        'prior_correction': 'none; raw case-control probabilities are not population-calibrated risk',
        'probability_aggregation': 'single context; same registered downstream held-meta calibration',
        'checkpoint_sha256': digest, 'vendor_inference_source_sha256': ENGINE_SHA256,
        'engine_audit': engine_audit, 'execution_backend': 'CUDA AMP float16',
        'seconds': time.time() - started,
        'context_selection_uses_query_labels': False,
        'context_selection_uses_query_covariates': False,
        'query_labels_used_for_early_stopping': False,
        'correction_labels_loaded': False, 'outer_labels_loaded': False,
        'feature_selection_changed': False, 'fitted_weights_saved': False, 'KV_caches_saved': False,
        'same_cache_replay_tolerance': 1e-7,
        'replay_scope': 'Original engine first-query-batch same-cache replay; not different-batch-size equivalence.',
    }


fit_predict = train_predict
