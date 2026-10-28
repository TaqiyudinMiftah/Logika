"""Bounded paired robustness pilots; immutable existing training roles and features."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import hashlib
import json
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import baseline as b
import feature_head
from gpu_backend import Engine

R = Path(__file__).resolve().parents[1]
A = R.parent
INPUT = A / 'ranking-pipeline-20261020/input'
MAIN = A / 'kumo-coverage-placement-20261021'
PACKETS = A / 'kumo-temperature-adaptation-20261021/full_data'
MATURITY = A / 'repayment-severity-fe-20261021'
PERIODIC = A / 'shopping-periodicity-fe-20261021'
REPLACEMENT = A / 'maturity-placement-fe-20261021'

def read(path):
    return json.loads(Path(path).read_text())

def check():
    assert time.time() < read(R / 'PLAN.json')['training_deadline_epoch']
    assert not (R / 'STOP').exists()
    for name, digest in read(R / 'PLAN.json')['source_sha256'].items():
        assert b.sha(R / 'source' / name) == digest, name

def inputs(fold):
    original = pd.read_parquet(INPUT / 'data/train_features.parquet', columns=['application_id', 'target'])
    ix = np.load(INPUT / f'artifacts/fold{fold}/indices.npz')
    packet = PACKETS / f'fold{fold}'
    for name, digest in read(packet / 'manifest.json')['files'].items():
        assert b.sha(packet / name) == digest, name
    q = pd.read_parquet(packet / 'query.parquet')
    context = pd.read_parquet(packet / 'context.parquet')
    yc = np.load(packet / 'correction_labels.npy').astype(int)
    source = MAIN / f'conditions/fold{fold}/control90'
    boundary = read(source / 'boundary.json')
    columns = boundary['feature_columns']
    assert len(columns) == 90 and 'target' not in q
    assert len(context) == 50000 and len(q) == 20000 and len(yc) == 6000
    assert q.application_id.tolist() == original.application_id.iloc[np.r_[ix['correction'], ix['validation']]].tolist()
    assert context.application_id.tolist() == original.application_id.iloc[ix['context']].tolist()
    assert np.array_equal(context.target.to_numpy(int), original.target.iloc[ix['context']].to_numpy(int))
    assert np.array_equal(yc, original.target.iloc[ix['correction']].to_numpy(int))
    assert set(context.application_id).isdisjoint(q.application_id)
    assert not np.intersect1d(ix['correction'], ix['validation']).size
    assert np.array_equal(np.sort(np.r_[ix['context'], ix['correction']]), np.sort(ix['train']))
    assert b.sha(source / 'raw_predictions.npz') == read(source / 'complete.json')['small_files']['raw_predictions.npz']
    raw = np.load(source / 'raw_predictions.npz')
    assert np.array_equal(raw['rows'], np.r_[ix['correction'], ix['validation']])
    head = joblib.load(source / 'correction.joblib')
    sp = joblib.load(MAIN / f'shared/fold{fold}/spline.joblib')
    hmat = joblib.load(MATURITY / f'heads/matched_maturity/fold{fold}/model.joblib')
    smat = joblib.load(REPLACEMENT / f'fold{fold}/spline.joblib')
    hper = joblib.load(PERIODIC / f'models/true/fold{fold}/head.joblib')
    sper = joblib.load(PERIODIC / f'models/true/fold{fold}/spline.joblib')
    needed = list(dict.fromkeys(sp['columns'] + sper['columns'] + head['columns'] + hper['columns']))
    available = pd.read_parquet(A / f'fold-local-priors-20261021/fold{fold}/features.parquet', columns=['application_id'] + sp['columns'])
    frame = available.set_index('application_id').loc[original.application_id].reset_index()
    spectral_cols = [c for c in needed if c.startswith('beh_spectral_')]
    spectral = pd.read_parquet(PERIODIC / 'true_features.parquet', columns=['application_id'] + spectral_cols).set_index('application_id').loc[frame.application_id]
    for c in spectral_cols:
        frame[c] = spectral[c].to_numpy()
        q[c] = spectral.loc[q.application_id, c].to_numpy()
    for c in sper['categorical']:
        frame[c] = frame[c].astype(object)
    mature = pd.read_parquet(MATURITY / f'fold{fold}/features.parquet').set_index('application_id')
    return original, ix, q, context, yc, columns, raw['prediction'], frame, mature, {
        'baseline': (head, sp), 'maturity': (hmat, smat), 'periodicity': (hper, sper)}

def stages(q, frame, mature, variant):
    query = q.copy()
    table = frame.copy()
    if variant != 'baseline':
        for threshold in (1, 30):
            col = f're_late_{threshold}_posterior'
            query[col] = mature.loc[query.application_id, f'matched_beta_late_{threshold}'].to_numpy()
            table[col] = mature.loc[table.application_id, f'matched_beta_late_{threshold}'].to_numpy()
    return query, table

def gate():
    check()
    cases = [([0, 0, 1, 1], [0, 1, 2, 3], 1.),
             ([0, 1, 0, 1], [.5, .5, .5, .5], 0.),
             ([0, 0, 1, 1], [0, 1, 1, 2], .5)]
    for y, p, expected in cases:
        y, p = np.array(y), np.array(p)
        assert abs(b.scores(y, p)['KS'] - expected) < 1e-12
    checks = []
    reference = np.load(PERIODIC / 'OOF.npz')
    for f in range(5):
        original, ix, q, context, yc, columns, raw, frame, mature, models = inputs(f)
        for variant, (head, sp) in models.items():
            query, table = stages(q, frame, mature, variant)
            pred = .75 * b.corrected_probability(head, query.iloc[6000:], raw[6000:])[0] + .25 * b.predict_spline(sp, table.iloc[ix['validation']])
            key = {'baseline': 'baseline', 'maturity': 'maturity', 'periodicity': 'true'}[variant]
            error = float(np.max(np.abs(pred - reference[key][ix['validation']])))
            assert error < 2e-14, (f, variant, error)
            checks.append(dict(fold=f, variant=variant, replay_error=error,
                               validation_positive=int(original.target.iloc[ix['validation']].sum())))
    checkpoint = A / 'ranking-pipeline-20261020/assets/classifier.pt'
    assert b.sha(checkpoint) == 'cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
    b.dump(dict(status='passed_for_bounded_paired_robustness_pilots', source_hashes=read(R/'PLAN.json')['source_sha256'],
                checks=checks, metric_cases_passed=True, role_boundaries_passed=True,
                inherited_snapshot_provenance_unresolved=True, independent_confirmation=False,
                broad_search_admitted=False), R / 'GATE.json')

def run(fold):
    check()
    assert read(R / 'GATE.json')['status'] == 'passed_for_bounded_paired_robustness_pilots'
    dest = R / f'robustness/fold{fold}'
    dest.mkdir(parents=True, exist_ok=False)
    started = time.time()
    original, ix, q, context, yc, columns, original_raw, frame, mature, models = inputs(fold)
    checkpoint = A / 'ranking-pipeline-20261020/assets/classifier.pt'
    seed = 12026 + fold
    engine = Engine(checkpoint, columns, context)
    raw = engine.fit_predict(np.arange(len(context)), q, seed, cache_path=None)
    assert engine.last_prediction_replay_error == 0.
    del engine
    np.savez_compressed(dest / 'raw_seed_repeat.npz', rows=np.r_[ix['correction'], ix['validation']], prediction=raw)
    checks = []
    for variant, (oldhead, sp) in models.items():
        query, table = stages(q, frame, mature, variant)
        spred = b.predict_spline(sp, table.iloc[ix['validation']])
        for mode in ['fresh_kumo_seed', 'correction_resample_51025', 'correction_resample_61025']:
            check()
            if mode == 'fresh_kumo_seed':
                rows = np.arange(6000)
                base = raw
            else:
                rng = np.random.default_rng(int(mode.rsplit('_', 1)[1]) + fold)
                rows = np.concatenate([rng.choice(np.flatnonzero(yc == c), int((yc == c).sum()), replace=True) for c in (0, 1)])
                base = original_raw
            training = query.iloc[rows].copy()
            training['target'] = yc[rows]
            head = feature_head.fit(training, base[:6000][rows], oldhead['columns'], .003, .5)
            pred = .75 * b.corrected_probability(head, query.iloc[6000:], base[6000:])[0] + .25 * spred
            path = dest / f'{variant}__{mode}.joblib'
            joblib.dump(head, path, compress=3)
            replay = .75 * b.corrected_probability(joblib.load(path), query.iloc[6000:], base[6000:])[0] + .25 * spred
            assert np.array_equal(pred, replay)
            np.savez_compressed(dest / f'{variant}__{mode}.npz', rows=ix['validation'], application_id=query.application_id.iloc[6000:].to_numpy(str), prediction=pred)
            checks.append(dict(variant=variant, mode=mode, head_sha256=b.sha(path), gradient=head['gradient_max_abs']))
            print(json.dumps(dict(stage='head_complete', fold=fold, variant=variant, mode=mode)), flush=True)
    b.dump(dict(status='complete', fold=fold, seed=seed, seconds=time.time()-started,
                context_source=str(PACKETS / f'fold{fold}/context.parquet'),
                context_source_sha256=b.sha(PACKETS / f'fold{fold}/context.parquet'),
                checkpoint_sha256=b.sha(checkpoint), canonical_context_retained_at_source=True,
                fitted_models=checks, source_sha256=read(R/'PLAN.json')['source_sha256'],
                no_validation_labels_passed_to_fit=True), dest/'complete.json')

def report():
    reference = np.load(PERIODIC / 'OOF.npz')
    y, folds = reference['target'], reference['fold']
    result = {}
    predictions = {}
    for variant, key in [('baseline','baseline'), ('maturity','maturity'), ('periodicity','true')]:
        original = reference[key]
        for mode in ['original', 'fresh_kumo_seed', 'correction_resample_51025', 'correction_resample_61025', 'two_seed_mean']:
            if mode == 'original':
                pred = original
            elif mode == 'two_seed_mean':
                pred = .5 * original + .5 * predictions[f'{variant}__fresh_kumo_seed']
            else:
                pred = np.full(len(y), np.nan)
                for f in range(5):
                    root = R / f'robustness/fold{f}'
                    assert read(root/'complete.json')['status'] == 'complete'
                    a = np.load(root / f'{variant}__{mode}.npz')
                    assert np.array_equal(a['rows'], np.flatnonzero(folds == f))
                    pred[a['rows']] = a['prediction']
            assert np.isfinite(pred).all()
            score = b.scores(y, pred)
            assert abs(score['KS'] - ks_2samp(pred[y==0], pred[y==1]).statistic) < 1e-12
            score['fold_KS'] = [b.scores(y[folds==f], pred[folds==f])['KS'] for f in range(5)]
            score['mean_fold_KS'] = float(np.mean(score['fold_KS']))
            result[f'{variant}__{mode}'] = score
            predictions[f'{variant}__{mode}'] = pred
    np.savez_compressed(R/'ROBUSTNESS_OOF.npz', target=y, fold=folds, **predictions)
    b.dump(dict(status='complete', metrics=result, interpretation='Conditional reused-development sensitivity; bootstrap correction samples and Kumo seeds are not new independent datasets.', independent_confirmation=False), R/'ROBUSTNESS_RESULT.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['gate', 'run', 'report'])
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    try:
        if args.action == 'gate': gate()
        elif args.action == 'run': run(args.fold)
        else: report()
    except BaseException:
        b.dump(dict(action=args.action, fold=args.fold, error=traceback.format_exc()), R/f'FAILURE_{args.action}_{args.fold}_{int(time.time())}.json')
        raise
