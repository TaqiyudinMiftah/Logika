"""H012/H013: train correction on an honestly predicted final mixture.

No new Kumo contexts, features, split choices, or validation-fitted thresholds.
The spline is fitted on context rows only so its correction-row outputs are OOS.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import argparse
import copy
import json
import time
import warnings
import joblib
import numpy as np
from scipy.optimize import minimize, check_grad
from scipy.special import logit
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import baseline as b
import feature_head
from robustness import R, A, INPUT, inputs, stages, read

D = R / 'joint_correction'
MODES = ('original', 'fresh_kumo_seed')
RECIPES = ('matched_control', 'final_mixture', 'balanced_final_mixture')


def check():
    plan = read(R / 'CONTINUATION_20261025.json')
    assert time.time() < plan['training_deadline_epoch']
    assert not (R / 'STOP').exists()
    for name, digest in read(R / 'H012_H013.json')['source_sha256'].items():
        assert b.sha(R / 'source' / name) == digest, name
    assert read(R / 'GATE.json')['status'] == 'passed_for_bounded_paired_robustness_pilots'
    assert read(R / 'H012_H013.json')['broad_search'] is False
    archive_bytes = sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    assert archive_bytes < plan['archive_total_limit_bytes'], archive_bytes
    return plan


def fit_spline(training, columns):
    assert len(training) in (50000, 64000)
    assert not set(columns) & {'target', 'customer_id', 'application_id', 'split'}
    cats = training[columns].select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    numeric = [c for c in columns if c not in cats]
    continuous = [c for c in numeric if training[c].nunique(dropna=True) > 10]
    linear = [c for c in numeric if c not in continuous]
    encoder = ColumnTransformer([
        ('spline', Pipeline([('fill', SimpleImputer(strategy='median')),
                            ('basis', SplineTransformer(n_knots=5, degree=3, knots='quantile', extrapolation='constant')),
                            ('scale', StandardScaler())]), continuous),
        ('linear', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), linear),
        ('missing', MissingIndicator(features='all', sparse=False, error_on_new=False), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)])
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        x = encoder.fit_transform(b.prepare(training, columns, cats))
        model = LogisticRegression(C=.05, solver='newton-cholesky', max_iter=80, tol=1e-5)
        model.fit(x, training.target.to_numpy(int), sample_weight=np.ones(len(training)))
    messages = [str(w.message) for w in captured]
    assert not any('converge' in m.lower() for m in messages), messages
    return dict(encoder=encoder, model=model, columns=columns, categorical=cats,
                fit_rows=len(training), fit_ids_sha=b.ids_hash(training, np.arange(len(training))),
                resolved_parameters=model.get_params(), warnings=messages)


def fit_balanced(ordinary, training, p):
    head = copy.deepcopy(ordinary)
    x, offset = b.design(head, training, p)
    y = training.target.to_numpy(int)
    assert len(y) == 6000 and set(np.unique(y)) == {0, 1}
    pi = float(y.mean())
    weights = np.where(y == 1, .5/pi, .5/(1-pi))
    # Fit a balanced posterior, then remove this known prior shift at prediction.
    # At the population optimum beta=0 when the supplied p is already calibrated.
    prior_shift = -float(logit(pi))
    penalty = .003
    with threadpool_limits(limits=2):
        small = np.r_[np.flatnonzero(y == 0)[:96], np.flatnonzero(y == 1)[:32]]
        args = (x[small], offset[small], y[small], weights[small], penalty, prior_shift)
        error = float(check_grad(lambda z:b.objective(z, *args)[0],
                                 lambda z:b.objective(z, *args)[1],
                                 np.zeros(x.shape[1]), direction='random', seed=2026102513))
        assert error < 2e-5, error
        result = minimize(b.objective, np.zeros(x.shape[1]),
                          args=(x, offset, y, weights, penalty, prior_shift),
                          method='L-BFGS-B', jac=True,
                          options=dict(maxiter=500, ftol=1e-11, gtol=1e-7))
        value, gradient = b.objective(result.x, x, offset, y, weights, penalty, prior_shift)
    assert result.success and np.max(abs(gradient)) < 1e-4, result.message
    head.update(coefficients=result.x, objective=value, gradient_max_abs=float(np.max(abs(gradient))),
                iterations=int(result.nit), positive_weight=float(.5/pi),
                balanced_objective=dict(training_prevalence=pi, fit_prior_shift=prior_shift,
                                        inference_prior_shift=0., derivative_error=error,
                                        fit_rows=len(y), purpose='Equal class contribution to logistic loss'))
    return head


def run(fold):
    check()
    dest = D / f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=False)
    started = time.time()
    original, ix, q, context, yc, columns, raw, frame, mature, models = inputs(fold)
    query, table = stages(q, frame, mature, 'maturity')
    headspec, oldsp = models['maturity']
    assert not np.intersect1d(ix['context'], np.r_[ix['correction'], ix['validation']]).size
    training = table.iloc[ix['context']].copy()
    training['target'] = original.target.iloc[ix['context']].to_numpy(int)
    sp = fit_spline(training, oldsp['columns'])
    joblib.dump(sp, dest / 'spline_context_only.joblib', compress=3)
    rows = np.r_[ix['correction'], ix['validation']]
    spline = b.predict_spline(sp, table.iloc[rows])
    replay_sp = b.predict_spline(joblib.load(dest / 'spline_context_only.joblib'), table.iloc[rows])
    assert np.array_equal(spline, replay_sp)
    correction = query.iloc[:6000].copy()
    correction['target'] = yc
    output = dict(rows=ix['validation'], application_id=query.application_id.iloc[6000:].to_numpy(str))
    replay_errors = []
    reference = np.load(R / 'ROBUSTNESS_OOF.npz')
    for mode in MODES:
        if mode == 'original':
            base = raw
            oldhead = headspec
        else:
            source = R / f'robustness/fold{fold}'
            complete = read(source / 'complete.json')
            assert complete['status'] == 'complete'
            base = np.load(source / 'raw_seed_repeat.npz')['prediction']
            oldhead = joblib.load(source / 'maturity__fresh_kumo_seed.joblib')
        old_replay = .75*b.corrected_probability(oldhead, query.iloc[6000:], base[6000:])[0] + .25*b.predict_spline(oldsp, table.iloc[ix['validation']])
        assert np.max(abs(old_replay-reference[f'maturity__{mode}'][ix['validation']])) < 2e-14
        mixture = .75*base + .25*spline
        ordinary = feature_head.fit(correction, mixture[:6000], headspec['columns'], .003, .5)
        balanced = fit_balanced(ordinary, correction, mixture[:6000])
        output[f'matched_control__{mode}'] = .75*b.corrected_probability(oldhead, query.iloc[6000:], base[6000:])[0] + .25*spline[6000:]
        for recipe, head in [('final_mixture', ordinary), ('balanced_final_mixture', balanced)]:
            path = dest / f'{recipe}__{mode}.joblib'
            head['training_roles'] = dict(spline_rows=50000, head_rows=6000,
                                           head_ids_sha=b.ids_hash(original, ix['correction']),
                                           spline_ids_sha=b.ids_hash(original, ix['context']),
                                           input_score='.75 raw_Kumo + .25 context_only_spline')
            joblib.dump(head, path, compress=3)
            prediction = b.corrected_probability(head, query.iloc[6000:], mixture[6000:])[0]
            replay = b.corrected_probability(joblib.load(path), query.iloc[6000:], mixture[6000:])[0]
            assert np.array_equal(prediction, replay)
            output[f'{recipe}__{mode}'] = prediction
            replay_errors.append(float(np.max(abs(prediction-replay))))
    np.savez_compressed(dest/'predictions.npz', **output)
    np.savez_compressed(dest/'spline_query.npz', rows=rows, prediction=spline)
    b.dump(dict(status='complete', fold=fold, seconds=time.time()-started,
                spline_fit_rows=50000, correction_fit_rows=6000, validation_rows=14000,
                all_fit_roles_disjoint=True, prior_scope='original50000 context applicants',
                source_plan_sha256=b.sha(R/'H012_H013.json'),
                original_baselines_replayed=True, saved_model_replay_max_error=max(replay_errors),
                files={p.name:b.sha(p) for p in dest.iterdir() if p.is_file()}), dest/'complete.json')
    print(json.dumps(dict(status='fold_complete', fold=fold, seconds=time.time()-started)), flush=True)


def weighted_ks(labels_sorted, counts_sorted, ends):
    positives = np.cumsum(counts_sorted*labels_sorted)
    negatives = np.cumsum(counts_sorted*(1-labels_sorted))
    return float(np.max(np.abs((negatives/negatives[-1]-positives/positives[-1])[ends])))


def report():
    assert not (D/'RESULT.json').exists()
    reference = np.load(R/'ROBUSTNESS_OOF.npz')
    y, folds = reference['target'], reference['fold']
    predictions = {f'{recipe}__{mode}':np.full(len(y), np.nan) for recipe in RECIPES for mode in MODES}
    checks = []
    for f in range(5):
        root = D/f'fold{f}'
        complete = read(root/'complete.json')
        assert complete['status']=='complete' and complete['source_plan_sha256']==b.sha(R/'H012_H013.json')
        for n,h in complete['files'].items(): assert b.sha(root/n)==h, n
        z = np.load(root/'predictions.npz')
        assert np.array_equal(z['rows'], np.flatnonzero(folds==f))
        for key in predictions: predictions[key][z['rows']]=z[key]
        checks.append(dict(fold=f, artifact_hashes_verified=True, model_replay=complete['saved_model_replay_max_error']))
    for recipe in RECIPES:
        predictions[f'{recipe}__two_seed_mean']=.5*(predictions[f'{recipe}__original']+predictions[f'{recipe}__fresh_kumo_seed'])
    metrics = {}
    for name,p in predictions.items():
        assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
        score = b.scores(y,p)
        assert abs(score['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-14
        score['fold_KS']=[b.scores(y[folds==f],p[folds==f])['KS'] for f in range(5)]
        metrics[name]=score
    comparisons={}; draws={}
    evaluated={k:predictions[f'{k}__two_seed_mean'] for k in RECIPES}
    evaluated['previous_B']=reference['maturity__original']
    sorted_info={}
    for name,p in evaluated.items():
        order=np.argsort(p,kind='stable'); s=p[order]
        sorted_info[name]=(order,y[order],np.r_[s[1:]!=s[:-1],True])
    for recipe in RECIPES[1:]:
        for control in ('matched_control','previous_B'): draws[f'{recipe}__{control}']=[]
    groups=[np.flatnonzero((folds==f)&(y==cl)) for f in range(5) for cl in (0,1)]
    rng=np.random.default_rng(202610251213)
    # Multinomial row counts are exactly the fixed-class/fold paired bootstrap.
    for _ in range(2000):
        counts=np.zeros(len(y),dtype=np.int64)
        for g in groups: counts[g]=rng.multinomial(len(g), np.full(len(g),1/len(g)))
        scores={name:weighted_ks(labels,counts[order],ends) for name,(order,labels,ends) in sorted_info.items()}
        for recipe in RECIPES[1:]:
            for control in ('matched_control','previous_B'): draws[f'{recipe}__{control}'].append(scores[recipe]-scores[control])
    eligible=[]
    basefold=[b.scores(y[folds==f],reference['maturity__original'][folds==f])['KS'] for f in range(5)]
    for recipe in RECIPES[1:]:
        name=f'{recipe}__two_seed_mean'; contrasts={}
        for control in ('matched_control','previous_B'):
            contrasts[control]=dict(point_delta=metrics[name]['KS']-b.scores(y,evaluated[control])['KS'],
                                    four_contrast_adjusted_95pct_interval=np.quantile(draws[f'{recipe}__{control}'],[.00625,.99375]).tolist())
        seed_gains={mode:metrics[f'{recipe}__{mode}']['KS']-metrics[f'matched_control__{mode}']['KS'] for mode in MODES}
        wins=sum(v>c for v,c in zip(metrics[name]['fold_KS'],basefold))
        passed=(metrics[name]['KS']>.5314098055476294 and wins>=3 and min(seed_gains.values())>0
                and all(c['four_contrast_adjusted_95pct_interval'][0]>0 for c in contrasts.values()))
        comparisons[recipe]=dict(contrasts=contrasts,seed_gains=seed_gains,fold_wins_vs_B=wins,promotion_passed=bool(passed))
        if passed: eligible.append(recipe)
    selected=max(eligible,key=lambda k:metrics[f'{k}__two_seed_mean']['KS']) if eligible else None
    np.savez_compressed(D/'OOF.npz',target=y,fold=folds,**predictions)
    np.savez_compressed(D/'BOOTSTRAP.npz',**draws)
    result=dict(status='complete',metrics=metrics,comparisons=comparisons,selected=selected,
                target_054_reached=any(metrics[f'{r}__two_seed_mean']['KS']>=.54 for r in RECIPES[1:]),
                checks=checks,plan_sha256=b.sha(R/'H012_H013.json'),independent_confirmation=False,
                limitation='Conditional reused-development comparisons; intervals exclude historical selection and refit variance.')
    b.dump(result,D/'RESULT.json')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['run','report'])
    parser.add_argument('--fold',type=int,choices=range(5))
    args=parser.parse_args()
    if args.action=='run': run(args.fold)
    else: report()
