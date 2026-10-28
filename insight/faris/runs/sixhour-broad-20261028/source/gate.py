"""Measured data admission and separate, bounded family runtime qualifications.

The data gate reads only the already-exposed matched anchor's outer labels.
Family gates never score a candidate against meta or outer labels. Every gate
attempt is retained, including missing evidence and failed pilot attempts.
"""
import os
for _key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_key] = '2'

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from common import R, load, dump, sha, ids_sha, utc, check_seal, tasks, catalog, metrics

FAMILIES = {'lightgbm', 'catboost', 'xgboost', 'glm', 'neural', 'kumo_large', 'kumo_medium'}
FORBIDDEN = {'target', 'application_id', 'customer_id', 'split', 'y_outer', 'y_validation', 'y_correction'}
REFERENCE_KS = .5306361042400267
FAMILY_PACKAGES = {
    'lightgbm': ['lightgbm'], 'catboost': ['catboost'], 'xgboost': ['xgboost'],
    'glm': ['scipy', 'scikit-learn'], 'neural': ['torch', 'tabm', 'rtdl-num-embeddings'],
    'kumo_large': ['torch', 'structured-data-models'], 'kumo_medium': ['torch', 'structured-data-models'],
}


def _evidence(path):
    path = Path(path)
    return {'path': str(path.relative_to(R)) if path.is_relative_to(R) else str(path), 'sha256': sha(path)}


def _private_error(kind, identifier):
    path = R / 'private/gate_errors' / f'{kind}_{identifier}_{time.time_ns()}.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(traceback.format_exc())
    return _evidence(path)


def _check(checks, kind, identifier, category, predicate, function, evidence=()):
    item = {'id': identifier, 'category': category, 'required': True,
            'acceptance_rule': predicate, 'evidence': []}
    try:
        item['observed'] = function()
        item['evidence'] = [_evidence(path) for path in evidence]
        item['status'] = 'pass'
    except FileNotFoundError:
        item.update(status='inconclusive', reason='Required input or completed-pilot evidence is missing.',
                    private_diagnostic=_private_error(kind, identifier))
    except BaseException:
        item.update(status='fail', reason='Measured evidence did not satisfy the declared predicate; inspect private diagnostic.',
                    private_diagnostic=_private_error(kind, identifier))
    checks.append(item)
    return item


def _finish(kind, canonical, clock, checks, extra=None):
    assert checks and all(item['required'] for item in checks)
    statuses = [item['status'] for item in checks]
    status = 'FAIL' if 'fail' in statuses else ('INCONCLUSIVE' if 'inconclusive' in statuses else 'PASS')
    folder = R / 'gate_attempts' / kind
    folder.mkdir(parents=True, exist_ok=True)
    number = 1 + max([int(p.name[7:]) for p in folder.glob('attempt*')
                      if p.is_dir() and p.name[7:].isdigit()] or [0])
    attempt = folder / f'attempt{number}'
    attempt.mkdir(exist_ok=False)
    result = {
        'status': status, 'scope': 'Adaptive reused-development admission; not untouched statistical confirmation.',
        'kind': kind, 'attempt': number, 'UTC': utc(), 'checks': checks,
        'clock_sha256': sha(R / 'CLOCK.json'), 'contract_sha256': sha(R / 'VALIDATION_CONTRACT.json'),
        'config_sha256': sha(R / 'CONFIG.json'), 'source_seal_sha256': sha(R / 'SOURCE_SEAL.json'),
        'gate_source_sha256': sha(__file__), 'independent_audit_available': False,
        'candidate_meta_or_outer_scores_computed': False,
    }
    previous = R / canonical
    if previous.exists():
        result['previous_gate_sha256'] = sha(previous)
        result['previous_gate_status'] = json.loads(previous.read_text()).get('status')
    if extra:
        result.update(extra)
    dump(result, attempt / 'RESULT.json')
    # A new pointer never edits an older gate attempt. A successful retry is
    # therefore explicit and cannot silently erase failed/inconclusive evidence.
    pointer = {**result, 'attempt_result': _evidence(attempt / 'RESULT.json')}
    dump(pointer, previous)
    print(json.dumps({'gate': kind, 'status': status, 'attempt': number}), flush=True)
    return pointer


def _valid_probability(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction, dtype=float)
    assert y.ndim == prediction.ndim == 1 and len(y) == len(prediction)
    assert np.isfinite(prediction).all() and np.all((prediction >= 0) & (prediction <= 1))
    assert set(np.unique(y)) == {0, 1}


def _scipy_ks(y, prediction):
    _valid_probability(y, prediction)
    return float(ks_2samp(prediction[y == 1], prediction[y == 0], method='asymp').statistic)


def evaluator_checks():
    cases = [
        (np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), 1.),
        (np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1]), 1.),
        (np.array([0, 1, 0, 1]), np.repeat(.5, 4), 0.),
        (np.array([0, 1, 0, 1]), np.array([.1, .3, .3, .8]), .5),
    ]
    scores = []
    for y, prediction, expected in cases:
        value = _scipy_ks(y, prediction)
        fpr, tpr, _ = roc_curve(y, prediction, drop_intermediate=False)
        assert abs(value - expected) < 1e-12
        assert abs(value - np.max(np.abs(tpr - fpr))) < 1e-12
        assert abs(value - metrics(y, prediction)['KS']) < 1e-12
        order = np.random.default_rng(20261029).permutation(len(y))
        assert abs(value - _scipy_ks(y[order], prediction[order])) < 1e-12
        scores.append(value)
    invalid_rejected = 0
    for bad in [np.array([np.nan, .2]), np.array([.2]), np.array([-.1, .5]), np.array([.2, np.inf])]:
        try:
            _valid_probability(np.array([0, 1]), bad)
        except AssertionError:
            invalid_rejected += 1
    assert invalid_rejected == 4
    return {'known_cases': scores, 'SciPy_ROC_and_campaign_KS_match': True,
            'tied_scores_and_joint_permutation_verified': True, 'invalid_outputs_rejected': 4}


def _parent_and_ids():
    inputs, references = load('INPUTS.json'), load('INPUT_REFERENCES.json')
    parent = Path(inputs['parent_root'])
    assert parent.resolve() != R.resolve()
    assert sha(parent / 'SOURCE_SEAL.json') == inputs['parent_source_seal_sha256']
    parent_gate = json.loads((parent / 'GATE.json').read_text())
    assert parent_gate['status'] == 'PASS'
    source = Path(references['target_source_file'])
    assert sha(source) == references['target_source_sha256']
    fields = pq.read_schema(source).names
    assert 'application_id' in fields
    selected = ['application_id'] + (['customer_id'] if 'customer_id' in fields else [])
    identity = pd.read_parquet(source, columns=selected)
    assert len(identity) == 70000 and identity['application_id'].notna().all()
    assert identity['application_id'].is_unique
    customer = 'Not present in inherited source; no new customer-group independence claim.'
    if 'customer_id' in identity:
        assert identity['customer_id'].notna().all() and identity['customer_id'].is_unique
        customer = 'Verified unique within the original70000 applicant rows.'
    return {'application_IDs': 70000, 'application_IDs_unique': True,
            'customer_identity': customer, 'source_file_sha256': sha(source),
            'parent_gate_sha256': sha(parent / 'GATE.json'),
            'parent_scope_retained': True,
            'availability_limit': 'Historical capture/cutoff completeness is inherited from the matched parent; no new external rows or labels.'}


def _preparation():
    prepared = load('PREPARATION_AUDIT.json')
    assert prepared['status'] == 'PASS'
    assert prepared['current_clock_sha256'] == sha(R / 'CLOCK.json')
    assert prepared['source_schema_sha256'] == sha(R / 'FEATURE_SCHEMA.json')
    assert prepared['source_bank_manifest_sha256'] == sha(R / 'BANK_MANIFEST.json')
    required = ['roles_disjoint', 'outer_rows_cover70000_once',
                'inner_training42500_excludes7500_stoppers', 'features_and_PCA_fit_only_inner_training']
    assert all(prepared[key] is True for key in required)
    assert prepared['outer_labels_in_worker_packets'] is False
    bank = load('BANK_MANIFEST.json')
    assert prepared['source_bank_hashes'] == {name: bank['files'][name]
        for name in ['numeric.npy', 'categorical.parquet', 'metadata.parquet']}
    schema = load('FEATURE_SCHEMA.json')
    assert len(schema['specs']) == len({item['id'] for item in schema['specs']})
    for item in schema['specs']:
        assert item['id'] not in FORBIDDEN and not set(item.get('inputs', [])) & FORBIDDEN
    return {'preparation_status': 'PASS', 'immutable_source_bank_hashes_match': True,
            'selector_and_PCA_boundary_statements_verified': required,
            'schema_definitions': len(schema['specs']), 'forbidden_source_inputs': 0}


def _packet(fold, target, baseline_folds):
    packet = R / f'packets/fold{fold}'
    manifest = json.loads((packet / 'MANIFEST.json').read_text())
    prepared = load('PREPARATION_AUDIT.json')
    assert sha(packet / 'MANIFEST.json') == prepared['packet_manifests'][str(fold)]
    for name, key in [('features.parquet', 'feature_sha256'), ('all_features.parquet', 'full_feature_sha256'),
                      ('roles.npz', 'roles_sha256'), ('inner_roles.npz', 'inner_roles_sha256'),
                      ('anchor.npz', 'anchor_sha256'), ('gate.parquet', 'gate_sha256')]:
        assert sha(packet / name) == manifest[key], name
    assert manifest['preparation_source_sha256'] == sha(R / 'source/prepare.py')
    assert manifest['source_schema_sha256'] == sha(R / 'FEATURE_SCHEMA.json')
    assert manifest['source_bank_manifest_sha256'] == sha(R / 'BANK_MANIFEST.json')
    assert manifest['original_clock_sha256'] == sha(R / 'CLOCK.json')
    role, inner = np.load(packet / 'roles.npz'), np.load(packet / 'inner_roles.npz')
    assert not {'y_outer', 'y_validation', 'target', 'labels'} & set(role.files)
    context, meta, outer = [role[key] for key in ['context', 'correction', 'validation']]
    assert (len(context), len(meta), len(outer)) == (50000, 6000, 14000)
    assert len(np.unique(np.r_[context, meta, outer])) == 70000
    assert np.array_equal(np.sort(outer), np.flatnonzero(baseline_folds == fold))
    assert np.array_equal(role['y_context'], target[context]) and np.array_equal(role['y_correction'], target[meta])
    train, stop = train_test_split(np.arange(50000), test_size=.15, stratify=role['y_context'], random_state=20261029 + fold)
    assert np.array_equal(inner['train_relative'], train) and np.array_equal(inner['stop_relative'], stop)
    assert np.array_equal(inner['inner_train'], context[train]) and np.array_equal(inner['inner_stop'], context[stop])
    assert np.array_equal(inner['y_inner'], role['y_context'][train]) and np.array_equal(inner['y_stop'], role['y_context'][stop])
    assert (len(train), len(stop)) == (42500, 7500)
    assert manifest['inner_split_seed'] == 20261029 + fold
    assert manifest['inner_train_rows_sha256'] == ids_sha(context[train])
    assert manifest['stopping_rows_sha256'] == ids_sha(context[stop])
    assert manifest['inner_train_labels_sha256'] == ids_sha(role['y_context'][train])
    assert manifest['stopping_labels_sha256'] == ids_sha(role['y_context'][stop])
    assert manifest['feature_statistics_fit_rows'] == 'inner_train42500'
    assert manifest['feature_statistics_exclude_stopping'] is True
    assert manifest['supervised_selection_excludes_stopping'] is True
    assert manifest['correction_or_outer_labels_used_for_feature_selection'] is False
    assert manifest['outer_labels_in_worker_packet'] is False
    assert manifest['fitted_selector_or_PCA_weights_saved'] is False
    selectors = {}
    for name in ['base', 'augmented']:
        selector = manifest['selector_parameters'][name]
        assert selector['training_rows'] == 42500
        assert selector['training_rows_sha256'] == ids_sha(context[train])
        assert selector['training_labels_sha256'] == ids_sha(role['y_context'][train])
        assert selector['correction_labels_used'] is False and selector['stopping_labels_used'] is False
        assert selector['outer_labels_used'] is False and selector['fitted_weights_saved'] is False
        selectors[name] = {'training_rows': 42500, 'parameters': selector['parameters']}
    pca = manifest['PCA_recipe']
    assert pca['training_rows_sha256'] == ids_sha(context[train])
    assert pca['training_rows'] == 42500 and not pca['query_rows_used'] and not pca['target_labels_used']
    assert not pca['components_saved']
    for filename, columns in [('features.parquet', manifest['feature_columns']),
                              ('all_features.parquet', manifest['full_feature_columns'])]:
        parquet = pq.ParquetFile(packet / filename)
        assert parquet.metadata.num_rows == 70000 and parquet.schema_arrow.names == columns
        assert not set(columns) & FORBIDDEN and len(columns) == len(set(columns))
    all_available = set(manifest['feature_columns']) | set(manifest['full_feature_columns'])
    for name, columns in manifest['views'].items():
        assert columns and len(columns) == len(set(columns)) and set(columns) <= all_available
        assert not set(columns) & FORBIDDEN
        if name.endswith('99') or name in ['global32', 'global48', 'global64', 'global80', 'pca64_core']:
            assert len(columns) <= 99
    support = {}
    for name, values, minimum in [('context', role['y_context'], 1000),
                                  ('inner_train', inner['y_inner'], 500), ('inner_stop', inner['y_stop'], 100),
                                  ('meta', role['y_correction'], 100), ('outer_anchor_only', target[outer], 100)]:
        count = np.bincount(np.asarray(values, int), minlength=2)
        assert len(count) == 2 and int(count.min()) >= minimum
        support[name] = count.tolist()
    return {'fold': fold, 'roles': [50000, 6000, 14000], 'inner_roles': [42500, 7500],
            'selector_training': selectors, 'class_support': support,
            'view_sizes': {key: len(value) for key, value in manifest['views'].items()},
            'PCA_and_prior_fit_boundary': '42500 inner-training rows; stopping/meta/outer excluded.'}


def _baseline_and_uncertainty(target, baseline, folds):
    _valid_probability(target, baseline)
    score = _scipy_ks(target, baseline)
    assert abs(score - REFERENCE_KS) < 1e-12
    assert abs(score - metrics(target, baseline)['KS']) < 1e-12
    assert _scipy_ks(target, np.repeat(float(np.mean(target)), len(target))) == 0.
    fold_scores = [_scipy_ks(target[folds == fold], baseline[folds == fold]) for fold in range(5)]
    order = np.argsort(baseline, kind='stable')
    ends = np.flatnonzero(np.r_[baseline[order][1:] != baseline[order][:-1], True])
    strata = [np.flatnonzero((folds == fold) & (target == label)) for fold in range(5) for label in [0, 1]]
    rng = np.random.default_rng(20261029)
    draws = []
    for _ in range(200):
        counts = np.bincount(np.concatenate([rng.choice(rows, len(rows), replace=True) for rows in strata]), minlength=70000)
        contribution = counts * (target / np.sum(target) - (1 - target) / np.sum(1 - target))
        draws.append(float(np.max(np.abs(np.cumsum(contribution[order])[ends]))))
    interval = np.quantile(draws, [.025, .975])
    assert np.isfinite(interval).all() and 0 <= interval[0] <= interval[1] <= 1
    return {'reference_KS': score, 'pooled_rows': 70000, 'trivial_constant_KS': 0.,
            'fold_KS': fold_scores, 'fold_KS_std': float(np.std(fold_scores, ddof=1)),
            'conditional_baseline_KS_95pct': interval.tolist(), 'bootstrap_repetitions': 200,
            'bootstrap_seed': 20261029, 'sampling_unit': 'Applications within original fold and class.',
            'precision_limitation': 'Conditional prediction uncertainty; overlapping-fold training and historical adaptive selection are not independent replications.'}


def data_gate():
    clock = check_seal()
    assert time.time() < clock['deadline_epoch']
    checks = []
    _check(checks, 'data', 'evaluator', 'evaluator',
           'Four known KS cases including reversed ranking/ties match SciPy, ROC and campaign evaluator within1e-12; invalid arrays rejected.',
           evaluator_checks, [R / 'source/common.py', Path(__file__)])
    _check(checks, 'data', 'parent_identity_availability', 'boundaries',
           'Original source/parent seal and prior gate match;70000 unique nonmissing application IDs; customer IDs unique when supplied.',
           _parent_and_ids, [R / 'INPUTS.json', R / 'INPUT_REFERENCES.json'])
    _check(checks, 'data', 'preparation_source_and_scope', 'boundaries',
           'Preparation PASS belongs to this clock/schema/bank and excludes query/stopping labels from feature fitting; no forbidden identifiers/targets are feature inputs.',
           _preparation, [R / 'PREPARATION_AUDIT.json', R / 'FEATURE_SCHEMA.json', R / 'BANK_MANIFEST.json'])
    try:
        target = np.load(R / 'evaluation_inputs/target.npy')
        base = np.load(R / 'evaluation_inputs/baseline.npz')
        baseline, folds = base['prediction'], base['fold']
        assert target.shape == baseline.shape == folds.shape == (70000,)
        assert set(np.unique(folds)) == set(range(5))
        prepared = load('PREPARATION_AUDIT.json')
        assert sha(R / 'evaluation_inputs/target.npy') == prepared['target_sha256']
        assert sha(R / 'evaluation_inputs/baseline.npz') == prepared['baseline_sha256']
        for fold in range(5):
            _check(checks, 'data', f'fold_{fold}_roles_features_support', 'boundaries_and_support',
                   'Exact sealed tables/roles,50000+6000+14000 exhaustive/disjoint;42500+7500 shared split and label hashes match selectors/PCA; adequate class support.',
                   lambda f=fold: _packet(f, target, folds), [R / f'packets/fold{fold}/MANIFEST.json'])
        _check(checks, 'data', 'matched_baseline_and_uncertainty', 'baseline_and_uncertainty',
               'Same original70000 rows/folds reproduce matched anchor KS0.5306361042400267 within1e-12; constant KS0; fold/conditional uncertainty finite.',
               lambda: _baseline_and_uncertainty(target, baseline, folds),
               [R / 'evaluation_inputs/target.npy', R / 'evaluation_inputs/baseline.npz'])
    except BaseException:
        checks.append({'id': 'baseline_inputs', 'category': 'baseline_and_uncertainty', 'required': True,
                       'acceptance_rule': 'Known reference target/baseline arrays match original row identity and preparation hashes.',
                       'status': 'inconclusive', 'reason': 'Reference inputs unavailable or invalid; no broad admission.',
                       'evidence': [], 'private_diagnostic': _private_error('data', 'baseline_inputs')})
    packet_hashes = load('PREPARATION_AUDIT.json').get('packet_manifests', {}) if (R / 'PREPARATION_AUDIT.json').exists() else {}
    return _finish('data', 'GATE.json', clock, checks, {'packet_manifests': packet_hashes,
        'new_candidate_outer_scores_opened': False, 'representativeness': 'Inherited applicant unit and cutoff assumptions; all70000 rows previously exposed development.'})


def family_gate(family):
    assert family in FAMILIES
    clock = check_seal()
    assert time.time() < clock['deadline_epoch']
    checks, chosen = [], {}
    def data_admission():
        gate = load('GATE.json')
        assert gate['status'] == 'PASS'
        for key, name in [('clock_sha256', 'CLOCK.json'), ('contract_sha256', 'VALIDATION_CONTRACT.json'),
                           ('source_seal_sha256', 'SOURCE_SEAL.json'), ('config_sha256', 'CONFIG.json')]:
            assert gate[key] == sha(R / name)
        return {'data_gate_status': 'PASS', 'data_gate_sha256': sha(R / 'GATE.json')}
    _check(checks, family, 'data_admission', 'boundaries',
           'Data gate PASS matches current immutable clock/config/contract/source seal.', data_admission, [R / 'GATE.json'])
    def pilot_identity():
        config = load('CONFIG.json')
        candidates = [key for key in config['pilot_tasks'] if catalog()[tasks()[key]['design']]['family'] == family]
        assert len(candidates) == 1
        task_id = candidates[0]
        task = tasks()[task_id]
        assert task['fold'] == 0 and task['seed'] == 2026
        folders = sorted((R / 'runs' / task_id).glob('attempt*'), key=lambda p: int(p.name[7:]))
        if not folders:
            raise FileNotFoundError('Declared runtime pilot has not produced an attempt')
        selected = folders[-1]
        if (selected / 'FAILURE.json').exists():
            raise RuntimeError('Latest declared pilot failed; preserve it before any execution retry')
        if not (selected / 'RESULT.json').exists():
            raise FileNotFoundError('Declared pilot is unfinished')
        assert sum((p / 'RESULT.json').exists() for p in folders) == 1, 'Multiple successful pilot attempts are not one unambiguous qualification'
        start = json.loads((selected / 'START.json').read_text())
        result = json.loads((selected / 'RESULT.json').read_text())
        assert start['pilot'] is True and result['status'] == 'complete'
        assert start['task'] == task and start['design'] == catalog()[task['design']]
        assert result['start_sha256'] == sha(selected / 'START.json')
        assert result['prediction_sha256'] == sha(selected / 'prediction.npz')
        assert 0 < float(result['seconds']) <= 900
        for key, name in [('clock_sha256', 'CLOCK.json'), ('contract_sha256', 'VALIDATION_CONTRACT.json'),
                           ('source_seal_sha256', 'SOURCE_SEAL.json'), ('config_sha256', 'CONFIG.json')]:
            assert start[key] == sha(R / name)
        assert start['source_sha256'] == sha(R / 'source/worker.py')
        assert not start['outer_labels_loaded'] and not start['correction_labels_loaded']
        assert not start['correction_labels_used_for_training'] and not start['query_labels_used_for_early_stopping']
        assert not result['weights_saved'] and not result['KV_cache_saved']
        chosen.update(task_id=task_id, folder=selected, start=start, result=result)
        return {'task_id': task_id, 'selected_attempt': selected.name, 'pilot_seconds': result['seconds'],
                'prior_failed_attempts': [_evidence(p / 'FAILURE.json') for p in folders[:-1] if (p / 'FAILURE.json').exists()],
                'logical_actual_pilot_configs': 1, 'pilot_parameters_changed': start['pilot_parameters_changed']}
    _check(checks, family, 'actual_predeclared_pilot_identity', 'runtime',
           'Exactly one successful declared fold0/seed2026 pilot, latest attempt complete within900seconds, exact START/prediction/source lineage; earlier failures explicitly retained.', pilot_identity)
    def outputs_and_backend():
        if not chosen:
            raise FileNotFoundError('Successful pilot identity required')
        folder, start, result = chosen['folder'], chosen['start'], chosen['result']
        packet = R / 'packets/fold0'
        role = np.load(packet / 'roles.npz')
        manifest = json.loads((packet / 'MANIFEST.json').read_text())
        query = np.r_[role['correction'], role['validation']]
        output = np.load(folder / 'prediction.npz')
        assert output['raw'].shape == (20000,) and np.array_equal(output['rows'], query)
        assert np.isfinite(output['raw']).all() and np.all((output['raw'] >= 0) & (output['raw'] <= 1))
        assert start['context_rows_sha256'] == ids_sha(role['context'])
        assert start['prediction_rows_sha256'] == ids_sha(query) and start['context_labels_sha256'] == ids_sha(role['y_context'])
        assert start['seed'] == 2026 and start['inner_split_seed'] == 20261029
        assert start['declared_inner_train_rows_sha256'] == manifest['inner_train_rows_sha256']
        assert start['declared_inner_stop_rows_sha256'] == manifest['stopping_rows_sha256']
        assert start['columns'] == manifest['views'][start['view']]
        assert start['packet_manifest_sha256'] == sha(packet / 'MANIFEST.json')
        audit = result['audit']
        if 'stopping_rows_sha256' in audit:
            assert audit['stopping_rows_sha256'] == manifest['stopping_rows_sha256']
        if 'inner_train_rows_sha256' in audit:
            assert audit['inner_train_rows_sha256'] == manifest['inner_train_rows_sha256']
        if family in {'lightgbm', 'catboost', 'xgboost', 'glm', 'neural'}:
            assert audit['refit_context_rows'] == 50000
            assert audit['correction_used_for_early_stopping'] is False
        needed = ['numpy', 'pandas', 'pyarrow', 'scikit-learn'] + FAMILY_PACKAGES[family]
        version = start['environment']
        assert all(key in version and version[key] and 'not installed' not in version[key] for key in needed)
        if start['design']['resource'] == 'gpu':
            assert start['device'] == 'cuda' and start['GPU_name'] and start['GPU_memory_bytes'] > 0
        if family.startswith('kumo'):
            size = 'medium' if family == 'kumo_medium' else 'large'
            checkpoint = load('CHECKPOINTS.json')[size]
            digest = checkpoint if isinstance(checkpoint, str) else checkpoint['sha256']
            assert audit['size'] == size and audit['checkpoint_sha256'] == digest
            assert audit['first_batch_replay_error'] < 1e-7 and len(start['columns']) <= 99
        chosen['dependency_versions'] = {key: version[key] for key in needed}
        return {'prediction_rows': 20000, 'context_rows': 50000, 'query_labels_used': False,
                'inner_stopping_rows': 7500, 'all_prediction_rows_exact': True,
                'versions': chosen['dependency_versions'], 'device': start['device'],
                'GPU_name': start.get('GPU_name'), 'pilot_scores_computed': False}
    _check(checks, family, 'runtime_predictions_boundaries_versions', 'runtime_and_boundaries',
           'Exact20000query rows,finite bounded predictions,context/label/inner-stop hashes,required package versions and declared CUDA backend; no meta/outer labels or fitted artifacts.', outputs_and_backend)
    extra = {'family': family, 'data_gate_sha256': sha(R / 'GATE.json') if (R / 'GATE.json').exists() else None,
             'dependency_versions': chosen.get('dependency_versions', {}),
             'qualification_scope': 'One predeclared actual runtime pilot; this is not predictive evidence for every future design/backend.'}
    if chosen:
        extra['pilot_result'] = _evidence(chosen['folder'] / 'RESULT.json')
        extra['pilot_start'] = _evidence(chosen['folder'] / 'START.json')
        extra['pilot_prediction'] = _evidence(chosen['folder'] / 'prediction.npz')
    return _finish(family, 'GATE_FAMILY_' + family + '.json', clock, checks, extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--data', action='store_true')
    choice.add_argument('--family', choices=sorted(FAMILIES))
    args = parser.parse_args()
    result = data_gate() if args.data else family_gate(args.family)
    raise SystemExit(0 if result['status'] == 'PASS' else 2)
