"""Post-result diagnostics of frozen predictions. Does not fit or select models."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent / 'exploration-12h-20261017'
DATA = ROOT.parents[2] / 'dataset'
OUT = ROOT / 'generalization-audit'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_statistic(labels, prediction):
    order = np.argsort(prediction, kind='stable')
    ends = np.r_[prediction[order][1:] != prediction[order][:-1], True]
    return order, ends


def weighted_ks(labels, weights, ordering):
    order, ends = ordering
    target = labels[order]
    weight = weights[order]
    a = np.cumsum(weight * target)
    b = np.cumsum(weight * (1 - target))
    return float(np.max(np.abs(a[ends] / a[-1] - b[ends] / b[-1])))


def paired_bootstrap(labels, candidate, champion, repetitions=2000, seed=20261018):
    """Resample identical applicants within each class for both predictions.

    Intervals condition on the already selected predictions and cannot undo
    historical/adaptive label reuse or cover full-refit variance.
    """
    a, b = ordered_statistic(labels, candidate), ordered_statistic(labels, champion)
    weight = np.ones(len(labels))
    difference = weighted_ks(labels, weight, a) - weighted_ks(labels, weight, b)
    indices = [np.flatnonzero(labels == c) for c in [0, 1]]
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions)
    for i in range(repetitions):
        sampled = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in indices])
        weights = np.bincount(sampled, minlength=len(labels))
        values[i] = weighted_ks(labels, weights, a) - weighted_ks(labels, weights, b)
    return {
        'delta_ks': difference,
        'conditional_paired_stratified_bootstrap_95ci': np.quantile(values, [.025, .975]).tolist(),
        'samples': repetitions,
        'seed': seed,
        'independent_confirmation': False,
        'caveat': 'Conditional on selected predictions; does not correct adaptive selection or full-refit variance.',
    }


def main():
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / 'train_applications.csv')
    test = pd.read_csv(DATA / 'test_applications.csv')
    evidence = ROOT / 'evidence'
    frozen = np.load(CAMPAIGN / 'evidence/champion_oof.npz')
    screen = np.load(evidence / 'selected_screen.npz')
    confirm = np.load(evidence / 'selected_validation.npz')
    assert np.array_equal(screen['rows'], frozen['rows'])
    assert not np.intersect1d(screen['rows'], confirm['rows']).size
    assert np.array_equal(np.sort(np.r_[screen['rows'], confirm['rows']]), np.arange(70000))
    metrics = {}
    for name, rows, candidate, champion in [
        ('development', screen['rows'], screen['prediction'], frozen['existing_feature_ensemble']),
        ('reused_validation', confirm['rows'], confirm['prediction'],
         np.load(CAMPAIGN / 'results-20261018/reports/champion_confirm.npy')),
    ]:
        y = train.target.to_numpy()[rows]
        metrics[name] = paired_bootstrap(y, candidate, champion)
        metrics[name]['rank_correlation'] = float(spearmanr(candidate, champion).statistic)
    old = pd.read_csv(evidence / 'previous_submission.csv').set_index('application_id')
    new = pd.read_csv(ROOT / 'artifacts/submission.csv').set_index('application_id')
    assert set(old.index) == set(new.index)
    test_correlation = float(spearmanr(new.default_probability, old.loc[new.index].default_probability).statistic)
    temporal = {}
    for name, frame in [('train', train), ('test', test)]:
        dates = pd.to_datetime(frame.decision_datetime)
        temporal[name] = {
            'minimum': str(dates.min()), 'maximum': str(dates.max()),
            'months': dates.dt.strftime('%Y-%m').value_counts().sort_index().to_dict(),
            'unique_customers': int(frame.customer_id.nunique()),
        }
    temporal['train_test_customer_overlap'] = len(set(train.customer_id) & set(test.customer_id))
    comparison = read(evidence / 'recipe_comparison.json')
    result = {
        'scope': 'Post-public-result diagnostics only; no training, new recipes, or submission changes.',
        'public_result': {
            'source': 'User report in this chat on 2026-10-18; upload identity not independently verified.',
            'champion': .56188, 'candidate': .55353, 'delta': .55353 - .56188,
            'candidate_sha256': sha(ROOT / 'artifacts/submission.csv'),
            'champion_sha256': sha(evidence / 'previous_submission.csv'),
        },
        'selection': {
            'final_candidates': len(comparison['evaluated']),
            'eligible': sum(r['eligible'] for r in comparison['evaluated']),
            'final_ranking_criterion': 'Highest reused-validation KS after development gates',
            'same_development_labels_reused_for_seed_and_partition_checks': True,
            'no_independent_generalization_gate_for_final_choice': True,
        },
        'uncertainty': metrics,
        'full_refit_test_candidate_champion_rank_correlation': test_correlation,
        'temporal_coverage': temporal,
        'artifact_integrity': read(ROOT / 'local_verification.json'),
        'historical_five_stream_uncertainty': read(CAMPAIGN / 'results-20261018/reports/confirmation.json'),
        'files': {str(p.relative_to(ROOT)): sha(p) for p in [
            evidence / 'recipe_comparison.json', evidence / 'selected_screen.npz',
            evidence / 'selected_validation.npz', ROOT / 'artifacts/submission.csv',
            ROOT / 'local_verification.json', ROOT / 'prepare.py',
        ]},
    }
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['public_result', 'selection', 'uncertainty',
        'full_refit_test_candidate_champion_rank_correlation']}, indent=2))


if __name__ == '__main__':
    main()
