"""Adaptive development scoreboard, with one frozen final conditional assessment.

Only complete two-seed/five-fold cohorts receive scores. Every new cohort's
prediction identity freezes before opening outer targets. Historical labels and
previous campaign outcomes are already exposed; no result is an untouched audit.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from common import R, load, dump, sha, ids_sha, utc, check_seal, catalog
from compose import MODES, freeze_design


def metrics(y, probability):
    y, probability = np.asarray(y, int), np.asarray(probability, float)
    assert y.shape == probability.shape and np.isfinite(probability).all()
    assert set(np.unique(y)) == {0, 1} and np.all((probability >= 0) & (probability <= 1))
    fpr, tpr, _ = roc_curve(y, probability, drop_intermediate=False)
    ks = float(np.max(np.abs(tpr - fpr)))
    independent = float(ks_2samp(probability[y == 1], probability[y == 0], method='asymp').statistic)
    assert abs(ks - independent) < 1e-12
    return dict(KS=ks, AUC=float(roc_auc_score(y, probability)),
                logloss=float(log_loss(y, probability, labels=[0, 1])), rows=len(y))


def _arrays(path, **arrays):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _event(value):
    with (R / 'EXPOSURES.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(UTC=utc(), **value)) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _prepare_cohort(design, mode, folds, baseline_hash, target_hash, cache=None):
    """Read no target values here; produce a complete frozen mean prediction."""
    config = load('CONFIG.json')
    cache = {} if cache is None else cache
    if design not in cache:
        design_frozen = freeze_design(design)
        if design_frozen is None:
            cache[design] = None
        else:
            all_seed_values, inputs = {name: [] for name in MODES}, {}
            for seed in config['seeds']:
                values = {name: np.full(70000, np.nan) for name in MODES}
                coverage = np.zeros(70000, dtype=np.uint8)
                for fold in range(5):
                    root = R / f'composed/{design}/s{seed}/fold{fold}'
                    result = json.loads((root / 'RESULT.json').read_text())
                    assert sha(root / 'prediction.npz') == result['prediction_sha256']
                    assert sha(root / 'FROZEN.json') == result['frozen_sha256']
                    data = np.load(root / 'prediction.npz')
                    assert data['rows'].dtype.kind in 'iu'
                    rows = np.asarray(data['rows'], int)
                    assert rows.shape == (14000,) and len(np.unique(rows)) == 14000
                    assert set(rows) == set(np.flatnonzero(folds == fold))
                    assert not np.any(coverage[rows])
                    coverage[rows] += 1
                    for name in MODES:
                        assert data[name].shape == (14000,) and np.isfinite(data[name]).all()
                        values[name][rows] = data[name]
                    inputs[str((root / 'RESULT.json').relative_to(R))] = sha(root / 'RESULT.json')
                assert np.all(coverage == 1)
                for name in MODES:
                    assert np.isfinite(values[name]).all()
                    all_seed_values[name].append(values[name])
            cache[design] = (design_frozen, all_seed_values, inputs)
    if cache[design] is None:
        return None
    design_frozen, all_seed_values, inputs = cache[design]
    seed_values = all_seed_values[mode]
    mean = np.mean(seed_values, axis=0)
    root = R / f'evaluation/cohorts/{design}__{mode}'
    identity = dict(design=design, mode=mode, design_frozen_sha256=sha(design_frozen),
                    composed_results=inputs, target_sha256=target_hash,
                    baseline_sha256=baseline_hash, evaluator_source_sha256=sha(__file__))
    if (root / 'FROZEN.json').exists():
        frozen = json.loads((root / 'FROZEN.json').read_text())
        assert frozen['identity'] == identity
        assert sha(root / 'prediction.npz') == frozen['prediction_sha256']
        data = np.load(root / 'prediction.npz')
        assert np.array_equal(data['prediction'], mean)
    else:
        root.mkdir(parents=True, exist_ok=True)
        _arrays(root / 'prediction.npz', rows=np.arange(70000, dtype=np.int32), prediction=mean)
        dump(dict(UTC=utc(), identity=identity, prediction_sha256=sha(root / 'prediction.npz'),
                  prediction_rows_sha256=ids_sha(np.arange(70000)), seeds=config['seeds'], folds=5,
                  aggregation='arithmetic mean of complete per-seed OOF probabilities, then pooled binary absolute KS',
                  outer_targets_opened_for_this_score=False,
                  statistically_independent_confirmation=False,
                  historical_exposure='All70k rows previously used; this is adaptive development.'), root / 'FROZEN.json')
    return root, mean, seed_values


def _score_cohort(root, mean, by_seed, y, baseline, folds):
    frozen = json.loads((root / 'FROZEN.json').read_text())
    identity = frozen['identity']
    if (root / 'RESULT.json').exists():
        result = json.loads((root / 'RESULT.json').read_text())
        assert result['frozen_sha256'] == sha(root / 'FROZEN.json')
        assert result['prediction_sha256'] == sha(root / 'prediction.npz')
        return result
    _event(dict(role='adaptive_development_cohort_score', rows=70000,
                design=identity['design'], mode=identity['mode'],
                frozen_sha256=sha(root / 'FROZEN.json'), independent_confirmation=False))
    reference = metrics(y, baseline)
    score = metrics(y, mean)
    fold_delta = [metrics(y[folds == fold], mean[folds == fold])['KS']
                  - metrics(y[folds == fold], baseline[folds == fold])['KS'] for fold in range(5)]
    seed_delta = [metrics(y, value)['KS'] - reference['KS'] for value in by_seed]
    fold_KS=[metrics(y[folds==f],mean[folds==f])['KS']for f in range(5)]
    result = dict(status='complete', UTC=utc(), design=identity['design'], mode=identity['mode'],
                  fold_KS=fold_KS,fold_mean_KS=float(np.mean(fold_KS)),fold_std_KS=float(np.std(fold_KS)),
                  metrics=score, delta_KS=score['KS'] - reference['KS'],
                  fold_delta_KS=fold_delta, seed_delta_KS=seed_delta,
                  positive_folds=sum(value > 0 for value in fold_delta),
                  positive_seeds=sum(value > 0 for value in seed_delta),
                  prediction_sha256=sha(root / 'prediction.npz'),
                  frozen_sha256=sha(root / 'FROZEN.json'),
                  development_gain_support='Unassessed until one final selection and conditional interval; no automatic promotion.',
                  target_054_point_reached=bool(score['KS'] >= .54),
                  independent_confirmation=False, weights_saved=False)
    dump(result, root / 'RESULT.json')
    return result


def paired_interval(y, selected, baseline, folds, clock):
    """Predeclared2000 conditional application bootstrap replicates by fold/class."""
    seed, repeats = 20261029, 2000
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero((folds == f) & (y == label)) for f in range(5) for label in [0, 1]]
    assert all(len(value) for value in strata)
    orders, endings = [], []
    for value in [selected, baseline]:
        order = np.argsort(value, kind='stable')
        orders.append(order)
        endings.append(np.flatnonzero(np.r_[value[order][1:] != value[order][:-1], True]))
    positive, negative = float(np.sum(y)), float(len(y) - np.sum(y))
    differences = []
    for repetition in range(repeats):
        if repetition % 25 == 0:
            assert time.time() < clock['deadline_epoch'] - 30, 'Original final-evaluation reserve exhausted'
        sample = np.concatenate([rng.choice(rows, len(rows), replace=True) for rows in strata])
        counts = np.bincount(sample, minlength=len(y))
        contribution = counts * (y / positive - (1 - y) / negative)
        scores = [float(np.max(np.abs(np.cumsum(contribution[order])[ends])))
                  for order, ends in zip(orders, endings)]
        differences.append(scores[0] - scores[1])
    return dict(delta_KS=metrics(y, selected)['KS'] - metrics(y, baseline)['KS'],
                percentile95=np.quantile(differences, [.025, .975]).tolist(),
                replicates=repeats, seed=seed, unit='applications resampled within fold/class',
                limitation='Conditional on fitted predictions; excludes refitting, adaptive candidate selection, historical reuse and public leaderboard uncertainty.')


def _final(board, candidates, y, baseline, folds, clock):
    selected = board['best_development']
    if selected is None:
        dump(dict(status='no_complete_cohort', UTC=utc(), target_054_reached=False,
                  primary_development_gain_supported=False, independent_confirmation=False,
                  submission_made=False, pending_designs=board['pending_designs']), R / 'FINAL_EVALUATION.json')
        return
    key = selected['cohort']
    root = R / f'evaluation/cohorts/{key}'
    identity = dict(cohort=key, cohort_result_sha256=sha(root / 'RESULT.json'),
                    cohort_frozen_sha256=sha(root / 'FROZEN.json'),
                    prediction_sha256=sha(root / 'prediction.npz'),
                    evaluator_source_sha256=sha(__file__),
                    selection_rule='largest complete-cohort development KS; ties by design/mode lexicographic order')
    path = R / 'FINAL_SELECTION.json'
    snapshot = R / 'evaluation/FINAL_SCOREBOARD_SNAPSHOT.json'
    if path.exists():
        final_selection = json.loads(path.read_text())
        assert final_selection['identity'] == identity, 'Final selection is immutable; never replace after seeing CI'
        assert sha(snapshot) == final_selection['source_scoreboard_sha256']
    else:
        assert not snapshot.exists(), 'Unreconciled final scoreboard snapshot exists'
        snapshot.write_bytes((R / 'SCOREBOARD.json').read_bytes())
        dump(dict(UTC=utc(), identity=identity, source_scoreboard_sha256=sha(snapshot),
                  source_scoreboard_snapshot=str(snapshot.relative_to(R)),
                  selection_uses_adaptive_development=True,
                  conditional_interval_not_yet_opened=True, independent_audit_available=False), path)
    if (R / 'FINAL_EVALUATION.json').exists():
        previous = load('FINAL_EVALUATION.json')
        assert previous.get('final_selection_sha256') == sha(path)
        return
    _event(dict(role='final_conditional_development_interval', rows=70000, selected_cohort=key,
                final_selection_sha256=sha(path), independent_confirmation=False))
    probability = np.load(root / 'prediction.npz')['prediction']
    interval = paired_interval(y, probability, baseline, folds, clock)
    result = candidates[key]
    prior=np.load(R/'evaluation_inputs/previous.npz')['prediction']
    previous_interval=paired_interval(y, probability, prior, folds, clock)
    previous_fold_delta=[metrics(y[folds==f],probability[folds==f])['KS']-metrics(y[folds==f],prior[folds==f])['KS']for f in range(5)]
    previous_seed_delta=[]
    design,mode=key.rsplit('__',1)
    for seed in load('CONFIG.json')['seeds']:
        seed_probability=np.full(70000,np.nan)
        for fold in range(5):
            data=np.load(R/f'composed/{design}/s{seed}/fold{fold}/prediction.npz');seed_probability[data['rows']]=data[mode]
        previous_seed_delta.append(metrics(y,seed_probability)['KS']-metrics(y,prior)['KS'])
    previous_support=previous_interval['delta_KS']>=.0005 and previous_interval['percentile95'][0]>0 and sum(v>0 for v in previous_fold_delta)>=4 and all(v>0 for v in previous_seed_delta)
    supported = bool(previous_support and result['delta_KS'] >= .0005 and interval['percentile95'][0] > 0
                     and result['positive_folds'] >= 4 and result['positive_seeds'] == len(load('CONFIG.json')['seeds']))
    final = dict(status='complete_adaptive_development_assessment', UTC=utc(),
                 selected_cohort=key, primary=key, primary_result=result,
                 primary_development_gain_supported=supported,
                 local_target_054_point_reached=bool(result['metrics']['KS'] >= .54),
                 local_target_054_and_support=bool(result['metrics']['KS'] >= .54 and supported),
                 uncertainty=interval, baseline=metrics(y, baseline),
                 previous_winner=metrics(y,prior),previous_winner_uncertainty=previous_interval,previous_winner_fold_delta=previous_fold_delta,previous_winner_seed_delta=previous_seed_delta,previous_winner_supported=bool(previous_support),
                 best_development=result, complete_cohorts=len(candidates),
                 complete_designs=board['complete_designs'], pending_designs=board['pending_designs'],
                 public_incumbent=dict(id='V25', retained_public_KS=.56822, current_leaderboard_lookup_performed=False),
                 independent_confirmation=False,
                 exposure='All70k examples and earlier campaign outcomes are adaptive reused development.',
                 final_selection_sha256=sha(path), prediction_sha256=sha(root / 'prediction.npz'),
                 source_scoreboard_sha256=sha(snapshot),
                 automatic_promotion=False, refit_performed=False, submission_made=False,
                 fitted_weights_saved=False, original_deadline_epoch=clock['deadline_epoch'])
    dump(final, R / 'FINAL_EVALUATION.json')
    dump(final, R / 'evaluation/RESULT.json')
    lines = ['# Twelve-hour base-model campaign: final adaptive development assessment', '',
             'Complete five-fold, two-seed cohorts only. No independent statistical audit exists.', '',
             '| Cohort | Pooled KS | Delta vs matched anchor |', '|---|---:|---:|']
    for name in sorted(candidates, key=lambda item: (-candidates[item]['metrics']['KS'], item)):
        item = candidates[name]
        lines.append(f"| {name} | {item['metrics']['KS']:.10f} | {item['delta_KS']:+.10f} |")
    lines.extend(['', f"Final selected development cohort: `{key}`.",
                  f"Conditional gain rule supported: {supported}. Local0.54 point reached: {final['local_target_054_point_reached']}.",
                  'No refit, Kaggle submission or public-score inference is made automatically.',
                  'Incomplete designs are execution limitations, not negative predictive evidence.'])
    (R / 'evaluation/RESULTS.md').write_text('\n'.join(lines) + '\n')


def main(final=False):
    clock = check_seal()
    assert load('GATE.json')['status'] == 'PASS', 'Development scoring requires a matching scientific gate'
    assert time.time() < clock['deadline_epoch'], 'Original campaign deadline has passed'
    lock = (R / 'evaluate.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = load('CONFIG.json')
    assert len(config['seeds']) == 2 and len(set(config['seeds'])) == 2
    baseline_file = np.load(R / 'evaluation_inputs/baseline.npz')
    baseline, folds = baseline_file['prediction'], baseline_file['fold']
    assert baseline.shape == folds.shape == (70000,) and set(np.unique(folds)) == set(range(5))
    assert np.isfinite(baseline).all()
    baseline_hash, target_hash = sha(R / 'evaluation_inputs/baseline.npz'), sha(R / 'evaluation_inputs/target.npy')
    ready, pending_designs, complete_designs, cache = {}, [], [], {}
    # Freeze every new complete aggregate before opening target.npy below.
    for design in catalog():
        design_ready = True
        for mode in MODES:
            prepared = _prepare_cohort(design, mode, folds, baseline_hash, target_hash, cache)
            if prepared is None:
                design_ready = False
                break
            ready[design + '__' + mode] = prepared
        (complete_designs if design_ready else pending_designs).append(design)
    previous = load('SCOREBOARD.json') if (R / 'SCOREBOARD.json').exists() else {}
    need_labels = bool(final or any(not (root / 'RESULT.json').exists() for root, _, _ in ready.values()))
    y = None
    if need_labels and ready:
        y = np.load(R / 'evaluation_inputs/target.npy')
        assert y.shape == (70000,) and set(np.unique(y)) == {0, 1}
    candidates = {}
    for name, (root, mean, by_seed) in ready.items():
        if y is None:
            result = json.loads((root / 'RESULT.json').read_text())
            assert result['frozen_sha256'] == sha(root / 'FROZEN.json')
            assert result['prediction_sha256'] == sha(root / 'prediction.npz')
        else:
            result = _score_cohort(root, mean, by_seed, y, baseline, folds)
        candidates[name] = result
    if candidates:
        key = min(candidates, key=lambda name: (-candidates[name]['metrics']['KS'], name))
        best = dict(cohort=key, **candidates[key])
    else:
        best = None
    reference = metrics(y, baseline) if y is not None else previous.get('baseline')
    board = dict(status='complete' if not pending_designs else 'running_with_partial_designs', UTC=utc(),
                 baseline=reference, candidates=candidates, results=candidates,
                 best_development=best, best_current=best,
                 complete_designs=complete_designs, pending_designs=pending_designs,
                 complete_cohorts=len(candidates), expected_modes=MODES,
                 aggregation='two-seed mean OOF probability, then pooled absolute binary KS',
                 all70000_rows_previously_exposed=True, independent_audit_available=False,
                 target_054_point_reached=bool(best and best['metrics']['KS'] >= .54),
                 development_support='Only assessed for the frozen final choice with2000 conditional replicates.',
                 public_incumbent=dict(id='V25', retained_public_KS=.56822),
                 automatic_promotion=False, submission_made=False)
    dump(board, R / 'SCOREBOARD.json')
    if final:
        if ready:
            assert y is not None
        _final(board, candidates, y, baseline, folds, clock)
    print(json.dumps(dict(stage='final_evaluation' if final else 'development_scoreboard',
                          complete_designs=len(complete_designs), pending_designs=len(pending_designs),
                          complete_cohorts=len(candidates),
                          best=None if best is None else dict(cohort=best['cohort'], KS=best['metrics']['KS'], delta_KS=best['delta_KS']))), flush=True)
    return board


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    arguments = parser.parse_args()
    main(final=arguments.final)
