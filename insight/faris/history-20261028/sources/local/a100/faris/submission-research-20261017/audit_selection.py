"""Read-only checks of the delivered candidate; writes an evidence report only."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BANK = ROOT.parent / 'depth-20261017/stage/bank'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ks(y, p):
    return float(ks_2samp(p[y == 0], p[y == 1]).statistic)


def fast_ks(y, p):
    order = np.argsort(p, kind='stable')
    labels, values = y[order], p[order]
    ends = np.r_[values[1:] != values[:-1], True]
    delta = np.cumsum(labels) / labels.sum() - np.cumsum(1-labels) / (1-labels).sum()
    return float(np.abs(delta[ends]).max())


def compare(y, candidate, reference, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    deltas = []
    for _ in range(1000):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(fast_ks(y[ix], candidate[ix]) - fast_ks(y[ix], reference[ix]))
    return {'candidate_ks': ks(y, candidate), 'reference_ks': ks(y, reference),
            'delta': ks(y, candidate)-ks(y, reference),
            'paired_bootstrap_95ci': np.quantile(deltas, [.025, .975]).tolist(),
            'bootstrap_samples': len(deltas),
            'caveat': 'Conditional on already-selected predictions; does not correct adaptive selection or historical label reuse.'}


def read_csv(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['application_id', 'default_probability']
        rows = list(reader)
    ids = [row['application_id'] for row in rows]
    p = np.array([float(row['default_probability']) for row in rows])
    assert len(ids) == len(set(ids)) == 30000
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return ids, p


def main():
    ids, p = read_csv(REPO / 'submission.csv')
    sample_ids, _ = read_csv(REPO / 'dataset/sample_submission(3).csv')
    assert ids == sample_ids
    saved = json.loads((ROOT / 'artifacts/complete.json').read_text())
    assert sha(REPO / 'submission.csv') == saved['submission_sha256']
    assert sha(ROOT / 'artifacts/submission.csv') == saved['submission_sha256']
    for folder, checksum in saved['models'].items():
        assert sha(ROOT / 'artifacts' / folder / 'pipeline.joblib') == checksum

    oof = np.load(ROOT / 'evidence/candidate_oof.npz')
    y = np.load(BANK / 'development_y.npy').astype(int)
    assert np.array_equal(oof['rows'], np.load(BANK / 'development_rows.npy'))
    assert np.array_equal(oof['fold'], np.load(BANK / 'folds.npy'))
    development = compare(y, oof['existing_feature_ensemble'], oof['kumo'], 20261018)
    cy = np.load(BANK / 'confirmation_y.npy').astype(int)
    cp = np.load(ROOT / 'artifacts/confirmation/primary/combined.npy')
    ref = np.load(ROOT / 'evidence/reference_confirmation.npy')
    confirmation = compare(cy, cp, ref, 20261019)
    old = json.loads((ROOT / 'artifacts/confirmation_report.json').read_text())
    assert abs(confirmation['candidate_ks'] - old['primary']['ks']) < 1e-12
    assert abs(confirmation['reference_ks'] - old['reference_v3_ks']) < 1e-12

    similarities = {}
    for name in ['submission_v3.csv', 'submission_v5.csv', 'submission_ks.csv']:
        old_ids, old_p = read_csv(REPO / 'Logika' / name)
        mapping = dict(zip(old_ids, old_p))
        aligned = np.array([mapping[i] for i in ids])
        similarities[name] = {'spearman': float(spearmanr(p, aligned).statistic),
                              'mean_absolute_probability_difference': float(np.abs(p-aligned).mean()),
                              'interpretation': 'Prediction similarity only; test labels are unavailable.'}
    report = {
        'submission_sha256': sha(REPO / 'submission.csv'),
        'csv_and_model_hash_checks_passed': True,
        'development_vs_new_kumo_recipe': development,
        'confirmation_vs_refitted_v3_recipe': confirmation,
        'test_prediction_similarity': similarities,
        'user_reported_best_public_score': 0.55845,
        'public_score_of_current_candidate': None,
        'limitations': [
            'The new Kumo OOF candidate differs from submission_kumo_large_logika_v3.csv.',
            'v3 reference refits all four components on campaign partitions, not the original mixed 10-fold/3-fold submission procedure.',
            'Current submission uses five all-label refits; campaign OOF uses single training-fold models, not repeated-seed final bagging.',
            'No matched current-protocol comparison against every listed historical model is available.',
            'Selecting among thousands of pipelines and blends creates optimism; bootstrap intervals do not remove it.',
            'Confirmation labels were historically used.',
            'Higher local KS does not establish higher public KS; v5 already illustrates this reversal.'
        ],
        'conclusion': 'Valid research candidate; superiority to the public-best v3 submission is not established.'
    }
    (ROOT / 'selection_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
