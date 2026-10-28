"""H016: one anchored smooth-Youden update to the existing 56k spline.

The correction, Kumo context, feature basis and 75/25 mixture stay fixed.
Saved bundles reference hash-verified original spline encoders and coefficients.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import argparse
import json
import shutil
import time
import joblib
import numpy as np
from scipy.optimize import minimize, check_grad
from scipy.special import expit, logit
from scipy.stats import ks_2samp
from threadpoolctl import threadpool_limits
import baseline as b
from robustness import R, PERIODIC, inputs, stages, read
from direct_ks import objective
from joint_correction import weighted_ks

D = R / 'spline_direct_ks'
MODES = ('original', 'fresh_kumo_seed', 'two_seed_mean')


def check():
    plan = read(R / 'CONTINUATION_20261025.json')
    assert time.time() < plan['training_deadline_epoch']
    assert not (R / 'STOP').exists()
    for name, digest in read(R / 'H016.json')['source_sha256'].items():
        assert b.sha(R / 'source' / name) == digest, name
    size = sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    assert size < plan['archive_total_limit_bytes'], size
    assert shutil.disk_usage(R).free > plan['minimum_archive_free_bytes']


def design(old, frame):
    with threadpool_limits(limits=2):
        basis = old['encoder'].transform(b.prepare(frame, old['columns'], old['categorical']))
        offset = old['model'].decision_function(basis)
    return np.column_stack([np.ones(len(frame)), basis]), offset


def predict(bundle, frame):
    assert b.sha(bundle['base_spline']) == bundle['base_sha256']
    old = joblib.load(bundle['base_spline'])
    x, offset = design(old, frame)
    return expit(offset + .5 * (x @ bundle['delta']))


def fit(old, training):
    assert len(training) == 56000
    x, offset = design(old, training)
    y = training.target.to_numpy(int)
    beta = np.zeros(x.shape[1])
    threshold = float(logit(y.mean()))
    args = (x, offset, y, beta, .5, threshold)
    small = np.r_[np.flatnonzero(y == 0)[:96], np.flatnonzero(y == 1)[:32]]
    test_args = (x[small], offset[small], y[small], beta, .5, threshold)
    with threadpool_limits(limits=2):
        derivative_error = float(check_grad(
            lambda z: objective(z, *test_args)[0],
            lambda z: objective(z, *test_args)[1], beta,
            direction='random', seed=2026102516))
        assert derivative_error < 2e-5, derivative_error
        initial = objective(beta, *args)[0]
        result = minimize(objective, beta, args=args, method='L-BFGS-B', jac=True,
                          options=dict(maxiter=300, ftol=1e-11, gtol=1e-7))
        final, gradient = objective(result.x, *args)
    assert result.success and np.max(abs(gradient)) < 1e-4, result.message
    assert final <= initial + 1e-10
    return result.x, dict(derivative_error=derivative_error,
        initial_objective=initial, final_objective=final,
        gradient_max_abs=float(np.max(abs(gradient))), iterations=int(result.nit),
        threshold=threshold, displacement_rms=float(np.sqrt(np.mean((.5*x@result.x)**2))))


def run(fold):
    check()
    root = D / f'fold{fold}'
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    original, ix, q, _, _, _, _, frame, mature, models = inputs(fold)
    _, table = stages(q, frame, mature, 'periodicity')
    old = models['periodicity'][1]
    base = PERIODIC / f'models/true/fold{fold}/spline.joblib'
    assert len(old['columns']) == 88
    assert not set(old['columns']) & {'target', 'customer_id', 'application_id', 'split'}
    training = table.iloc[ix['train']].copy()
    training['target'] = original.target.iloc[ix['train']].to_numpy(int)
    assert len(np.intersect1d(ix['train'], ix['validation'])) == 0
    delta, optimizer = fit(old, training)
    bundle = dict(base_spline=str(base), base_sha256=b.sha(base), delta=delta,
                  training_ID_hash=b.ids_hash(original, ix['train']),
                  validation_ID_hash=b.ids_hash(original, ix['validation']),
                  fit_rows=56000, source_sha256=b.sha(__file__),
                  hypothesis_sha256=b.sha(R/'H016.json'), optimizer=optimizer)
    joblib.dump(bundle, root/'spline_update.joblib', compress=3)
    val = table.iloc[ix['validation']]
    new = predict(bundle, val)
    assert np.array_equal(new, predict(joblib.load(root/'spline_update.joblib'), val))
    old_p = b.predict_spline(old, val)
    # Store a single branch delta. All full pipeline outputs are reconstructed
    # from hash-verified frozen reference OOF during the report.
    np.savez_compressed(root/'validation.npz', rows=ix['validation'], spline_delta=new-old_p)
    b.dump(dict(status='complete', fold=fold, seconds=time.time()-started,
                source_sha256=b.sha(__file__), hypothesis_sha256=b.sha(R/'H016.json'),
                optimizer=optimizer, files={p.name:b.sha(p) for p in root.iterdir() if p.is_file()}),
           root/'complete.json')
    print(json.dumps(dict(stage='complete', fold=fold, seconds=time.time()-started)), flush=True)


def report():
    assert not (D/'RESULT.json').exists()
    plan = read(R/'H016.json')
    assert b.sha(R/'ROBUSTNESS_OOF.npz') == plan['reference_sha256']
    ref = np.load(R/'ROBUSTNESS_OOF.npz')
    y, folds = ref['target'], ref['fold']
    shift = np.full(len(y), np.nan)
    for f in range(5):
        root = D/f'fold{f}'
        complete = read(root/'complete.json')
        assert complete['status'] == 'complete'
        for name, digest in complete['files'].items(): assert b.sha(root/name) == digest
        z = np.load(root/'validation.npz')
        assert np.array_equal(z['rows'], np.flatnonzero(folds == f))
        shift[z['rows']] = .25*z['spline_delta']
    assert np.isfinite(shift).all()
    controls = {mode:ref[f'periodicity__{mode}'] for mode in MODES}
    predictions = {mode:controls[mode]+shift for mode in MODES}
    metrics = {}
    info = {}
    for kind, ps in [('candidate', predictions), ('control', controls)]:
        for mode, p in ps.items():
            assert np.isfinite(p).all() and p.min() > 0 and p.max() < 1
            score = b.scores(y, p)
            assert abs(score['KS']-ks_2samp(p[y == 0],p[y == 1]).statistic) < 1e-14
            score['fold_KS'] = [b.scores(y[folds == f],p[folds == f])['KS'] for f in range(5)]
            metrics[f'{kind}__{mode}'] = score
            order = np.argsort(p, kind='stable'); sorted_p = p[order]
            info[f'{kind}__{mode}'] = (order, y[order], np.r_[sorted_p[1:] != sorted_p[:-1], True])
    rng = np.random.default_rng(20261025160)
    groups = [np.flatnonzero((folds == f)&(y == cl)) for f in range(5) for cl in (0,1)]
    draws = {mode:[] for mode in MODES}
    for _ in range(2000):
        counts = np.zeros(len(y), dtype=int)
        for g in groups: counts[g] = rng.multinomial(len(g), np.full(len(g), 1/len(g)))
        values = {name:weighted_ks(labels,counts[order],ends) for name,(order,labels,ends) in info.items()}
        for mode in MODES: draws[mode].append(values[f'candidate__{mode}']-values[f'control__{mode}'])
    comparisons = {}
    for mode in MODES:
        c, old = metrics[f'candidate__{mode}'], metrics[f'control__{mode}']
        comparisons[mode] = dict(KS_delta=c['KS']-old['KS'],
            three_comparison_adjusted95_interval=np.quantile(draws[mode],[.025/3,1-.025/3]).tolist(),
            fold_wins=sum(a>b for a,b in zip(c['fold_KS'],old['fold_KS'])))
    primary = metrics['candidate__two_seed_mean']['KS']
    passed = primary > .5314098055476294 and comparisons['two_seed_mean']['fold_wins'] >= 3 and all(
        c['KS_delta'] > 0 and c['three_comparison_adjusted95_interval'][0] > 0 for c in comparisons.values())
    np.savez_compressed(D/'OOF.npz',target=y,fold=folds,**predictions)
    np.savez_compressed(D/'BOOTSTRAP.npz',**draws)
    result = dict(status='complete',metrics=metrics,comparisons=comparisons,
                  selected='two_seed_mean' if passed else None,promotion_passed=bool(passed),
                  target_054_reached=primary>=.54,hypothesis_sha256=b.sha(R/'H016.json'),
                  independent_confirmation=False,
                  qualification='Fixed reused development; conditional bootstrap excludes adaptive selection and refit variance.')
    b.dump(result,D/'RESULT.json')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','report'])
    parser.add_argument('--fold',type=int,choices=range(5));args=parser.parse_args()
    if args.action=='run': run(args.fold)
    else: report()
