"""Verified packets and downstream recipe, without serialized estimators."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import baseline as b
import feature_head

A = Path('/mnt/home/faris/logika-faris-exploration')
INPUT = A / 'ranking-pipeline-20261020/input'
PACKETS = A / 'kumo-temperature-adaptation-20261021/full_data'
MAIN = A / 'kumo-coverage-placement-20261021'

def read(p):
    return json.loads(p.read_text())

def checked_npz(root, name, digest):
    assert b.sha(root/name) == digest, root/name
    return np.load(root/name)

def inputs(fold):
    original = pd.read_parquet(INPUT/'data/train_features.parquet', columns=['application_id', 'target'])
    ix = np.load(INPUT/f'artifacts/fold{fold}/indices.npz')
    packet = PACKETS/f'fold{fold}'
    manifest = read(packet/'manifest.json')
    for name, digest in manifest['files'].items():
        assert b.sha(packet/name) == digest, name
    q = pd.read_parquet(packet/'query.parquet')
    context = pd.read_parquet(packet/'context.parquet')
    yc = np.load(packet/'correction_labels.npy').astype(int)
    source = MAIN/f'conditions/fold{fold}/control90'
    boundary = read(source/'boundary.json')
    cols, head_cols = boundary['feature_columns'], boundary['head_columns']
    assert len(cols) == 90 and len(head_cols) == 18 and 'target' not in q
    assert len(context) == 50000 and len(q) == 20000 and len(yc) == 6000
    assert q.application_id.tolist() == original.application_id.iloc[np.r_[ix['correction'], ix['validation']]].tolist()
    assert context.application_id.tolist() == original.application_id.iloc[ix['context']].tolist()
    assert np.array_equal(context.target, original.target.iloc[ix['context']])
    assert np.array_equal(yc, original.target.iloc[ix['correction']])
    assert set(context.application_id).isdisjoint(q.application_id)
    assert len(np.unique(np.r_[ix['context'], ix['correction'], ix['validation']])) == len(original)
    mroot = A/f'repayment-severity-fe-20261021/fold{fold}'
    assert b.sha(mroot/'features.parquet') == read(mroot/'transformer.json')['feature_hash']
    mature = pd.read_parquet(mroot/'features.parquet').set_index('application_id')
    query = q.copy()
    for threshold in (1, 30):
        query[f're_late_{threshold}_posterior'] = mature.loc[q.application_id, f'matched_beta_late_{threshold}'].to_numpy()
    # Recover the unchanged mature spline from hash-verified retained predictions.
    # both - head_only = .25 * (new_spline - old_spline).
    base_sp = np.load(packet/'spline_validation.npz')
    root = A/f'maturity-placement-fe-20261021/fold{fold}'
    z = checked_npz(root, 'validation.npz', read(root/'complete.json')['files']['validation.npz'])
    assert np.array_equal(base_sp['rows'], ix['validation']) and np.array_equal(z['rows'], ix['validation'])
    spline = base_sp['prediction'] + 4*(z['both']-z['head_only'])
    assert np.isfinite(spline).all() and spline.min() >= -1e-12 and spline.max() <= 1+1e-12
    raw = checked_npz(source, 'raw_predictions.npz', read(source/'complete.json')['small_files']['raw_predictions.npz'])
    assert np.array_equal(raw['rows'], np.r_[ix['correction'], ix['validation']])
    return original, ix, q, context, yc, cols, query, head_cols, spline, raw['prediction'], z['both']

def gate():
    checks = []
    for fold in range(5):
        original, ix, q, context, yc, cols, query, head_cols, spline, raw, expected = inputs(fold)
        head = feature_head.fit(query.iloc[:6000].assign(target=yc), raw[:6000], head_cols, .003, .5)
        replay = .75*b.corrected_probability(head, query.iloc[6000:], raw[6000:])[0] + .25*spline
        error = float(abs(replay-expected).max())
        assert error < 2e-12, (fold, error)
        checks.append({'fold': fold, 'max_abs_replay_error': error, 'head_features': len(head_cols), 'foundation_features': len(cols)})
    return {'status': 'passed', 'checks': checks, 'no_fitted_models_loaded_or_saved': True,
            'spline_reconstruction': 'hash-verified old_spline + 4*(both-head_only)',
            'independent_confirmation': False}

