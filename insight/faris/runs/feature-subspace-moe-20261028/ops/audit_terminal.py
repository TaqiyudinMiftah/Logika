"""Independent artifact readback; this never changes selection or refits a model."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import check_seal, dump, ids_sha, load, sha, utc
from compose import completed

def independent_ks(y, probability):
    return float(ks_2samp(probability[y == 1], probability[y == 0], method='asymp').statistic)

def conditional_interval(y, p, baseline, folds):
    rng = np.random.default_rng(20261028)
    strata = [np.flatnonzero((folds == f) & (y == label)) for f in range(5) for label in [0, 1]]
    order = [np.argsort(value, kind='stable') for value in [p, baseline]]
    ends = [np.flatnonzero(np.r_[value[index][1:] != value[index][:-1], True])
            for value, index in zip([p, baseline], order)]
    values = []
    for repetition in range(2000):
        assert time.time() < load('CLOCK.json')['deadline_epoch'] - 60
        sample = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        counts = np.bincount(sample, minlength=len(y))
        positive = counts * (y == 1)
        negative = counts * (y == 0)
        scores = []
        for index, ending in zip(order, ends):
            cdf_positive = np.cumsum(positive[index]) / positive.sum()
            cdf_negative = np.cumsum(negative[index]) / negative.sum()
            scores.append(float(np.max(np.abs(cdf_positive[ending] - cdf_negative[ending]))))
        values.append(scores[0] - scores[1])
    return np.quantile(values, [.025, .975])

def main():
    clock = check_seal()
    assert time.time() < clock['deadline_epoch'] - 1200
    config, frozen, evaluation = load('CONFIG.json'), load('FROZEN.json'), load('evaluation/RESULT.json')
    assert frozen['primary'] == config['primary'] == evaluation['primary']
    assert not frozen['selection_after_outer_scores']
    assert evaluation['frozen_sha256'] == sha(R / 'FROZEN.json')
    assert evaluation['status'] == 'complete', 'Incomplete execution is not a complete scientific comparison'
    for file, expected in frozen['composed_results'].items():
        assert sha(R / file) == expected
    target = np.load(R / 'evaluation_inputs/target.npy')
    baseline_file = np.load(R / 'evaluation_inputs/baseline.npz')
    baseline, folds = baseline_file['prediction'], baseline_file['fold']
    assert target.shape == baseline.shape == folds.shape == (70000,)
    assert abs(independent_ks(target, baseline) - .5306361042400267) < 1e-12
    packets = {}
    for fold in range(5):
        root = R / f'packets/fold{fold}'
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        assert sha(root / 'MANIFEST.json') == frozen['source_manifests'][str(fold)]
        assert sha(root / 'features.parquet') == manifest['feature_sha256']
        assert sha(root / 'roles.npz') == manifest['roles_sha256']
        role = np.load(root / 'roles.npz')
        context, meta, outer = role['context'], role['correction'], role['validation']
        assert (len(context), len(meta), len(outer)) == (50000, 6000, 14000)
        assert len(np.unique(np.r_[context, meta, outer])) == 70000
        assert set(outer) == set(np.flatnonzero(folds == fold))
        assert 'y_validation' not in role.files and 'y_outer' not in role.files
        assert np.array_equal(role['y_context'], target[context])
        packets[fold] = (manifest, role)
    backend_hashes, results = set(), {}
    for task, definition in config['tasks'].items():
        folder = completed(task)
        assert folder is not None
        start = json.loads((folder / 'START.json').read_text())
        end = json.loads((folder / 'RESULT.json').read_text())
        assert start['task'] == definition and end['task_id'] == task
        assert sha(folder / 'START.json') == end['start_sha256']
        assert start['source_sha256'] == sha(R / 'source/worker.py')
        assert start['source_seal_sha256'] == sha(R / 'SOURCE_SEAL.json')
        assert start['seed'] == definition['seed'] + definition['fold']
        assert start['contract_sha256'] == clock['contract_sha256']
        assert not start['outer_labels_loaded'] and not start['correction_labels_used']
        assert not end['weights_saved'] and not end['KV_cache_saved']
        manifest, role = packets[definition['fold']]
        query = np.r_[role['correction'], role['validation']]
        assert start['columns'] == manifest['views'][definition['view']]
        assert start['context_rows_sha256'] == ids_sha(role['context'])
        assert start['prediction_rows_sha256'] == ids_sha(query)
        assert start['context_labels_sha256'] == ids_sha(role['y_context'])
        prediction = np.load(folder / 'prediction.npz')
        assert np.array_equal(prediction['rows'], query)
        assert prediction['raw'].shape == (20000,) and np.isfinite(prediction['raw']).all()
        if definition['family'] == 'kumo':
            backend = json.loads((folder / 'EXECUTION_BACKEND.json').read_text())
            assert folder.name == 'attempt2' and backend['execution_backend'] == 'Colab account2 A100'
            assert backend['checkpoint_sha256'] == 'cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
            assert backend['original_clock_sha256'] == sha(R / 'CLOCK.json')
            assert not backend['scientific_parameters_changed']
            backend_hashes.add((backend['adapter_sha256'], backend['environment_sha256']))
        else:
            _, stop = train_test_split(np.arange(50000), test_size=.15,
                                       stratify=role['y_context'], random_state=start['seed'])
            assert ids_sha(role['context'][stop]) == end['audit']['stopping_rows_sha256']
            assert end['audit']['refit_context_rows'] == 50000
            assert not end['audit']['correction_used_for_early_stopping']
        results[task] = sha(folder / 'prediction.npz')
    assert len(results) == 260 and len(backend_hashes) == 1
    saved = np.load(R / 'evaluation/predictions.npz')
    for name, report in evaluation['results'].items():
        family, condition, combination = name.split('__')
        by_seed = []
        for seed in config['seeds']:
            value = np.full(70000, np.nan)
            for fold in range(5):
                root = R / f'composed/{family}/{condition}/s{seed}/fold{fold}'
                record = json.loads((root / 'RESULT.json').read_text())
                assert not record['outer_labels_loaded'] and not record['weights_saved']
                prediction = np.load(root / 'prediction.npz')
                assert np.array_equal(prediction['rows'], packets[fold][1]['validation'])
                assert sha(root / 'prediction.npz') == record['prediction_sha256']
                value[prediction['rows']] = prediction[combination]
            assert np.isfinite(value).all()
            by_seed.append(value)
        mean = np.mean(by_seed, axis=0)
        assert np.array_equal(mean, saved[name])
        assert abs(independent_ks(target, mean) - report['metrics']['KS']) < 1e-12
        seed_delta = [independent_ks(target, value) - independent_ks(target, baseline) for value in by_seed]
        fold_delta = [independent_ks(target[folds == f], mean[folds == f]) - independent_ks(target[folds == f], baseline[folds == f]) for f in range(5)]
        assert np.allclose(seed_delta, report['seed_delta_KS'], atol=1e-12, rtol=0)
        assert np.allclose(fold_delta, report['fold_delta_KS'], atol=1e-12, rtol=0)
    primary = evaluation['primary_result']
    interval = conditional_interval(target, saved[config['primary']], baseline, folds)
    assert np.allclose(interval, primary['uncertainty']['percentile95'], atol=1e-12, rtol=0)
    supported = bool(primary['delta_KS'] >= .0005 and interval[0] > 0 and
                     sum(value > 0 for value in primary['fold_delta_KS']) >= 4 and
                     all(value > 0 for value in primary['seed_delta_KS']))
    assert supported == evaluation['primary_development_gain_supported']
    dump({'status': 'PASS', 'UTC': utc(), 'scope': 'Artifact and arithmetic consistency, not independent statistical confirmation.',
          'source_seal_sha256': sha(R / 'SOURCE_SEAL.json'), 'clock_sha256': sha(R / 'CLOCK.json'),
          'frozen_sha256': sha(R / 'FROZEN.json'), 'evaluation_sha256': sha(R / 'evaluation/RESULT.json'),
          'expert_fits_verified': 260, 'Kumo_backend_cohorts': 1, 'seed_and_fold_boundaries_verified': True,
          'all_cohort_scores_recomputed_with_SciPy_KS': True, 'primary_conditional_interval_recomputed': interval.tolist(),
          'baseline_recomputed_KS': independent_ks(target, baseline),
          'primary_acceptance_rule_recomputed': supported, 'independent_audit_available': False,
          'scientific_selection_changed': False}, R / 'ARTIFACT_CONSISTENCY_AUDIT.json')

if __name__ == '__main__':
    main()
