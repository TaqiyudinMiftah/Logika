"""Read-only terminal evidence audit. Never chooses a new model or trains an architecture."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def epoch(value):
    return dt.datetime.fromisoformat(value).timestamp()


def independent_ks(y, p):
    # Group equal scores first, then compare the two empirical CDFs.
    _, inverse = np.unique(p, return_inverse=True)
    pos = np.bincount(inverse, weights=np.asarray(y) == 1)
    neg = np.bincount(inverse, weights=np.asarray(y) == 0)
    return float(np.max(np.abs(np.cumsum(pos) / pos.sum() - np.cumsum(neg) / neg.sum())))


def independent_auc(y, p):
    positive = np.asarray(y) == 1
    n1 = int(positive.sum())
    n0 = len(y) - n1
    return float((rankdata(p)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(args):
    root, data = Path(args.root), Path(args.frozen)
    config = read(root / 'CONFIGS.json')
    clock, gate = read(root / 'CLOCK.json'), read(root / 'GATE.json')
    selected = read(root / 'evaluation/FROZEN.json')
    result = read(root / 'evaluation/RESULT.json')
    launch = read(root / 'LAUNCH_AUDIT.json')
    seal = read(root / 'FEATURE_FREEZE.json')
    audit = read(root / 'MODEL_RUN_AUDIT.json')
    for filename, key in [('FEATURE_FREEZE.json', 'feature_freeze_sha256'),
                          ('CONFIGS.json', 'configs_sha256'),
                          ('VALIDATION_CONTRACT.json', 'contract_sha256')]:
        assert digest(root / filename) == clock[key] == gate[key] == selected[key]
    assert digest(root / 'CLOCK.json') == launch['clock_sha256']
    assert digest(root / 'evaluation/OOF.npz') == result['OOF_sha256']
    assert digest(root / 'evaluation/FROZEN.json') == result['frozen_recipe_sha256']
    assert digest(root / 'source/evaluate.py') == selected['source_sha256']
    for filename, expected in launch['source_sha256'].items():
        assert digest(root / 'source' / filename) == expected
    assert len(config) == 32 and len(selected['complete_designs']) == 16
    expected = {(name, f) for name in config for f in range(5)}
    assert set(map(tuple, audit['completed'])) == expected
    assert not audit['failed'] and not audit['pending'] and not audit['unresolved']
    assert not list(root.glob('runs/*/fold*/FAILURE.json'))
    assert len(list(root.glob('runs/*/fold*/RESULT.json'))) == 160
    forbidden = [p for p in root.rglob('*') if p.is_file() and
                 p.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors', '.pkl', '.joblib'}]
    assert not forbidden, forbidden
    roles, manifests, baselines, raw, stats = {}, {}, {}, {}, []
    coverage = np.zeros(70000, dtype=np.int32)
    source_worker, source_arch = digest(root / 'source/worker.py'), digest(root / 'source/architectures.py')
    for f in range(5):
        folder = data / f'fold{f}'
        manifests[f] = m = read(folder / 'MANIFEST.json')
        assert digest(folder / 'MANIFEST.json') == seal['folds'][f]['manifest_sha256']
        for filename, key in [('features.parquet', 'features_sha256'),
                              ('roles.npz', 'roles_sha256'), ('baseline.npz', 'baseline_sha256')]:
            assert digest(folder / filename) == m[key]
        roles[f], baselines[f] = np.load(folder / 'roles.npz'), np.load(folder / 'baseline.npz')
        role = roles[f]
        all_rows = np.r_[role['context'], role['correction'], role['validation']]
        assert len(np.unique(all_rows)) == 70000 and set(all_rows) == set(range(70000))
        assert [len(role[x]) for x in ['context', 'correction', 'validation']] == [50000, 6000, 14000]
        coverage[role['validation']] += 1
        train, stop = train_test_split(np.arange(50000), test_size=.1,
                                      stratify=role['y_context'], random_state=261027 + f)
        for name, spec in config.items():
            p = root / 'runs' / name / f'fold{f}'
            start, done, training = read(p / 'START.json'), read(p / 'RESULT.json'), read(p / 'TRAINING.json')
            assert start['config'] == spec and start['config_id'] == name and start['fold'] == f
            assert start['seed'] == spec['seed'] + f and start['pilot'] is False
            assert start['host'] in ['labkc-comm1', 'labkc-comm3']
            assert start['source_sha256'] == source_worker and start['architecture_source_sha256'] == source_arch
            assert start['clock_sha256'] == digest(root / 'CLOCK.json')
            assert start['feature_freeze_sha256'] == clock['feature_freeze_sha256']
            assert start['features_sha256'] == m['features_sha256'] and start['roles_sha256'] == m['roles_sha256']
            assert start['columns'] == m['views'][spec['view']]
            assert start['preprocessing']['fit_rows_count'] == 50000
            assert not start['correction_labels_used_for_training'] and not start['outer_labels_loaded']
            assert not done['outer_labels_loaded'] and not done['fitted_weights_saved'] and not training['weights_saved']
            assert (start['training_rows'], start['inner_stop_rows'], start['refit_rows']) == (45000, 5000, 50000)
            for key, rows in [('context_rows_sha256', role['context']),
                              ('inner_train_rows_sha256', role['context'][train]),
                              ('inner_stop_rows_sha256', role['context'][stop])]:
                assert start[key] == hashlib.sha256(rows.tobytes()).hexdigest()
            inner, refit = training['internal_stop_trace'], training['refit_trace']
            best, chosen = -np.inf, None
            for item in inner:
                assert np.isfinite(item['training_loss']) and np.isfinite(item['internal_stop_KS'])
                if item['internal_stop_KS'] > best + 1e-6:
                    best, chosen = item['internal_stop_KS'], item['epoch']
            assert chosen == done['selected_epochs'] == training['selected_epochs'] == len(refit)
            assert spec['min_epochs'] <= len(inner) <= spec['max_epochs']
            assert all(np.isfinite(item['training_loss']) for item in refit)
            assert epoch(start['UTC']) >= clock['started_epoch']
            assert epoch(done['UTC']) < epoch(selected['UTC']) < epoch(result['UTC']) < clock['deadline_epoch']
            assert done['seconds'] < 2100 and done['prediction_repeat_max_abs'] < 1e-6
            assert done['status'] == 'complete' and done['device'] == start['device']
            assert digest(p / 'prediction.npz') == done['prediction_sha256']
            pred = np.load(p / 'prediction.npz')
            assert np.array_equal(pred['rows'], np.r_[role['correction'], role['validation']])
            raw[name, f] = pred['raw']
            assert pred['raw'].shape == (20000,) and np.isfinite(pred['raw']).all()
            assert 0 <= pred['raw'].min() <= pred['raw'].max() <= 1
            stats.append({'name': name, 'fold': f, 'host': start['host'], 'device': start['device'],
                          'seconds': done['seconds'], 'epochs': chosen, 'max_RSS_KiB': done['max_RSS_KiB'],
                          'peak_CUDA_bytes': done['peak_CUDA_bytes']})
    assert np.all(coverage == 1)
    oof, reference = np.load(root / 'evaluation/OOF.npz'), np.load(args.reference)
    y, folds, baseline = oof['target'], oof['fold'], oof['baseline']
    assert np.array_equal(y, reference['target']) and np.array_equal(folds, reference['fold'])
    assert np.array_equal(baseline, reference['maturity__original'])
    assert set(np.unique(y)) == {0, 1} and set(np.unique(folds)) == set(range(5))
    raw_variants = {name: np.full(70000, np.nan) for name in result['diagnostic_metrics']}
    max_selection_error = 0.
    for f in range(5):
        role, base = roles[f], baselines[f]['raw']
        assert np.array_equal(y[role['context']], role['y_context'])
        assert np.array_equal(y[role['correction']], role['y_correction'])
        assert np.all(folds[role['validation']] == f)
        ranking, choice = train_test_split(np.arange(6000), test_size=.5,
                                          stratify=role['y_correction'], random_state=261026)
        pairs = {d: np.mean([raw[name, f] for name in names.values()], axis=0)
                 for d, names in selected['complete_designs'].items()}
        order = sorted(pairs, key=lambda d: (-independent_ks(role['y_correction'][ranking], pairs[d][ranking]), d))
        recipe = selected['recipes'][f]
        best = None
        for trial in recipe['trials']:
            count, w = trial['members'], trial['weight']
            values = (1-w)*base + w*np.mean([pairs[d] for d in order[:count]], axis=0)
            score = independent_ks(role['y_correction'][choice], values[choice])
            max_selection_error = max(max_selection_error, abs(score - trial['choice_KS']))
            key = (score, -w, -count)
            if best is None or key > best[0]:
                best = (key, values, order[:count], w)
        assert best[2] == recipe['designs'] and best[3] == recipe['weight']
        saved_primary = np.load(root / f'evaluation/fold{f}_primary.npz')
        assert np.array_equal(saved_primary['raw'], best[1])
        candidates = {name: raw[name, f] for name in config}
        candidates.update(pairs, primary=best[1], baseline_replay=base)
        for seed in ['2026', '7731']:
            member = np.mean([raw[selected['complete_designs'][d][seed], f] for d in recipe['designs']], axis=0)
            candidates['primary_seed'+seed] = (1-recipe['weight'])*base + recipe['weight']*member
        for name, values in candidates.items():
            raw_variants[name][role['validation']] = values[6000:]
    assert max_selection_error < 1e-12
    max_metric_error = 0.
    for name, recorded in result['diagnostic_metrics'].items():
        p = oof[name]
        assert p.shape == (70000,) and np.isfinite(p).all() and 0 <= p.min() <= p.max() <= 1
        actual = {'KS': independent_ks(y, p), 'raw_KS': independent_ks(y, raw_variants[name]),
                  'AUC': independent_auc(y, p), 'delta_KS': independent_ks(y, p)-independent_ks(y, baseline),
                  'fold_delta': [independent_ks(y[folds == f], p[folds == f])-
                                 independent_ks(y[folds == f], baseline[folds == f]) for f in range(5)]}
        for key in actual:
            max_metric_error = max(max_metric_error, float(np.max(np.abs(np.asarray(actual[key])-recorded[key]))))
    assert max_metric_error < 1e-12
    replay_error = float(np.max(abs(oof['baseline_replay']-baseline)))
    assert replay_error < 1e-10
    print(json.dumps({'stage': 'provenance_selection_metrics_passed', 'fits': len(stats),
                      'max_metric_error': max_metric_error, 'max_selection_error': max_selection_error}), flush=True)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    # Independent grouping-based weighted CDF bootstrap; identical predeclared random draws.
    inverse = [np.unique(oof[k], return_inverse=True)[1] for k in ['primary', 'baseline']]
    rng = np.random.default_rng(271026)
    delta = []
    for _ in range(2000):
        draws = np.r_[rng.choice(positive, len(positive), replace=True),
                      rng.choice(negative, len(negative), replace=True)]
        w = np.bincount(draws, minlength=len(y))
        scores = []
        for group in inverse:
            pos = np.bincount(group, weights=w*(y == 1))
            neg = np.bincount(group, weights=w*(y == 0))
            scores.append(np.max(abs(np.cumsum(pos)/len(positive)-np.cumsum(neg)/len(negative))))
        delta.append(scores[0]-scores[1])
    ci = np.quantile(delta, [.025, .975])
    assert np.max(abs(ci-result['uncertainty']['percentile95'])) < 1e-10
    head_error = None
    if args.head_source:
        import pandas as pd
        sys.path.insert(0, str(Path(args.head_source).resolve()))
        import baseline as b
        import feature_head
        head_error = 0.
        for f in range(5):
            role = roles[f]
            table = pd.read_parquet(data/f'fold{f}/features.parquet').iloc[np.r_[role['correction'],role['validation']]].reset_index(drop=True)
            # Replay every diagnostic, the primary and fixed-seed recipes from preserved raw outputs.
            candidates = {name: raw[name, f] for name in config}
            for d, members in selected['complete_designs'].items():
                candidates[d] = np.mean([raw[name, f] for name in members.values()], axis=0)
            candidates.update(primary=np.load(root/f'evaluation/fold{f}_primary.npz')['raw'],
                              baseline_replay=baselines[f]['raw'])
            recipe = selected['recipes'][f]
            for seed in ['2026','7731']:
                members = np.mean([raw[selected['complete_designs'][d][seed], f] for d in recipe['designs']], axis=0)
                candidates['primary_seed'+seed] = (1-recipe['weight'])*baselines[f]['raw']+recipe['weight']*members
            for name, values in candidates.items():
                head = feature_head.fit(table.iloc[:6000].assign(target=role['y_correction']), values[:6000], manifests[f]['head_columns'], .003, .5)
                pred = .75*b.corrected_probability(head, table.iloc[6000:], values[6000:])[0]+.25*baselines[f]['spline']
                head_error = max(head_error, float(np.max(abs(pred-oof[name][role['validation']]))))
            print(json.dumps({'stage':'head_replay', 'fold':f, 'max_abs_error':head_error}),flush=True)
        assert head_error < 1e-10
    finished = epoch(result['UTC'])
    report = {'status': 'passed', 'UTC': dt.datetime.now(dt.timezone.utc).isoformat(),
              'fits_verified': len(stats), 'configs': len(config), 'paired_designs': len(selected['complete_designs']),
              'failure_count': 0, 'OOF_rows': len(y), 'exact_once_outer_coverage': bool(np.all(coverage == 1)),
              'immutable_feature_tables_verified': 5, 'clock_preserved': True, 'source_and_config_seals_verified': True,
              'training_roles_seeds_columns_verified': True, 'frozen_selection_independently_reproduced': True,
              'all_training_finished_before_selection': True, 'selection_frozen_before_terminal_scoring': True,
              'all_fits_finished_within_budget': True, 'weights_or_KV_caches_serialized': False,
              'metric_variants_verified': len(result['diagnostic_metrics']), 'max_metric_error': max_metric_error,
              'max_selection_score_error': max_selection_error, 'baseline_replay_max_abs': replay_error,
              'head_replay_max_abs': head_error, 'bootstrap_replicates': 2000, 'recomputed_CI95': ci.tolist(),
              'host_device_counts': dict(collections.Counter(x['host']+'/'+x['device'] for x in stats)),
              'fit_seconds_sum': sum(x['seconds'] for x in stats), 'longest_fit_seconds': max(x['seconds'] for x in stats),
              'max_fit_RSS_GiB': max(x['max_RSS_KiB'] for x in stats)/1024**2,
              'max_CUDA_GiB': max(x['peak_CUDA_bytes'] for x in stats)/1024**3,
              'actual_campaign_seconds': finished-clock['started_epoch'], 'ceiling_seconds': clock['hours']*3600,
              'primary_KS': independent_ks(y, oof['primary']), 'baseline_KS': independent_ks(y, baseline),
              'target_reached': result['target_reached'], 'development_gain_supported': result['development_gain_supported'],
              'independent_confirmation': False, 'scientific_limit': 'Reused development evidence, not untouched confirmation.',
              'head_sources': {str(p):digest(p) for p in [Path(args.head_source)/'baseline.py',Path(args.head_source)/'feature_head.py']} if args.head_source else {},
              'audit_source_sha256':digest(__file__), 'evidence_sha256':{name:digest(root/name) for name in ['CLOCK.json','CONFIGS.json','VALIDATION_CONTRACT.json','FEATURE_FREEZE.json','GATE.json','MODEL_RUN_AUDIT.json','evaluation/RESULT.json','evaluation/FROZEN.json','evaluation/OOF.npz']}}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--frozen', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--head-source')
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
