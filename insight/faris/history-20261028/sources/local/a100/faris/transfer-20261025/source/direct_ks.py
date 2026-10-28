"""One fixed, anchored smooth-Youden correction objective; no parameter sweep."""
from pathlib import Path
import argparse, copy, json, time
import numpy as np
import joblib
from scipy.optimize import minimize, check_grad
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
import baseline as b
from robustness import R, A, MAIN, PERIODIC, inputs, check, read

TEMPERATURE = .5
TRUST = 1.
RIDGE = .003

def objective(delta, x, offset, y, beta, strength, threshold):
    displacement = strength * (x @ delta)
    z = (offset + strength * (x @ beta) + displacement - threshold) / TEMPERATURE
    u = expit(z)
    a = np.where(y == 1, -1 / int(y.sum()), 1 / int((y == 0).sum()))
    value = float(a @ u + .5 * TRUST * np.mean(displacement ** 2) + .5 * RIDGE * (delta @ delta))
    grad = strength * (x.T @ (a * u * (1-u) / TEMPERATURE + TRUST * displacement / len(y))) + RIDGE * delta
    return value, grad

def fit(old, training, p):
    x, offset = b.design(old, training, p)
    y = training.target.to_numpy(int)
    assert len(y) == 6000 and set(np.unique(y)) == {0, 1}
    beta = old['coefficients'].copy()
    strength = old['spec']['correction_strength']
    threshold = float(logit(y.mean()))
    args = (x, offset, y, beta, strength, threshold)
    with threadpool_limits(limits=2):
        # A numerical derivative on a bounded, deterministic training subset
        # verifies the new objective implementation before the optimizer runs.
        small = np.r_[np.flatnonzero(y==0)[:96], np.flatnonzero(y==1)[:32]]
        smallargs = (x[small], offset[small], y[small], beta, strength, threshold)
        delta0 = np.zeros_like(beta)
        error = float(check_grad(lambda z:objective(z,*smallargs)[0], lambda z:objective(z,*smallargs)[1], delta0, direction='random', seed=20261025))
        assert error < 2e-5, error
        initial = objective(delta0, *args)[0]
        result = minimize(objective, delta0, args=args, method='L-BFGS-B', jac=True,
                          options=dict(maxiter=300, ftol=1e-11, gtol=1e-7))
        value, gradient = objective(result.x, *args)
    assert result.success and np.max(np.abs(gradient)) < 1e-4, result.message
    assert value <= initial + 1e-10
    head = copy.deepcopy(old)
    head['coefficients'] = beta + result.x
    head['direct_KS'] = dict(temperature=TEMPERATURE, trust=TRUST, ridge=RIDGE,
                             threshold=threshold, fit_rows=6000, gradient_max=float(np.max(np.abs(gradient))),
                             derivative_check_error=error, initial_objective=initial, final_objective=value,
                             logit_displacement_rms=float(np.sqrt(np.mean((strength*x@result.x)**2))),
                             objective_source_sha256=b.sha(Path(__file__)))
    return head

def run(fold):
    check()
    assert read(R/'GATE.json')['status'] == 'passed_for_bounded_paired_robustness_pilots'
    dest = R/f'direct_ks/fold{fold}'
    dest.mkdir(parents=True, exist_ok=False)
    original, ix, q, context, yc, columns, raw, frame, mature, models = inputs(fold)
    sp = models['baseline'][1]
    spline = b.predict_spline(sp, frame.iloc[ix['validation']])
    training = q.iloc[:6000].copy()
    training['target'] = yc
    start = time.time()
    for mode in ['original', 'fresh_kumo_seed']:
        if mode == 'original':
            oldhead = models['baseline'][0]
            p = raw
        else:
            root = R/f'robustness/fold{fold}'
            assert read(root/'complete.json')['status'] == 'complete'
            oldhead = joblib.load(root/'baseline__fresh_kumo_seed.joblib')
            p = np.load(root/'raw_seed_repeat.npz')['prediction']
        head = fit(oldhead, training, p[:6000])
        model = dest/f'{mode}.joblib'
        joblib.dump(head, model, compress=3)
        def predict(h):
            return .75 * b.corrected_probability(h, q.iloc[6000:], p[6000:])[0] + .25 * spline
        pred = predict(head)
        assert np.array_equal(pred, predict(joblib.load(model)))
        np.savez_compressed(dest/f'{mode}.npz', rows=ix['validation'], application_id=q.application_id.iloc[6000:].to_numpy(str), prediction=pred)
    b.dump(dict(status='complete', fold=fold, seconds=time.time()-start, training_rows=6000,
                inference_rows=14000, fit_roles_disjoint=True,
                files={p.name:b.sha(p)for p in dest.iterdir()if p.is_file()}), dest/'complete.json')

def report():
    z=np.load(PERIODIC/'OOF.npz')
    y,fold=z['target'],z['fold']
    result={}
    predictions={}
    for mode in ['original','fresh_kumo_seed']:
        pred=np.full(len(y),np.nan)
        for f in range(5):
            root=R/f'direct_ks/fold{f}'
            assert read(root/'complete.json')['status']=='complete'
            a=np.load(root/f'{mode}.npz')
            assert np.array_equal(a['rows'],np.flatnonzero(fold==f))
            pred[a['rows']]=a['prediction']
        assert np.isfinite(pred).all()
        predictions[mode]=pred
    predictions['two_seed_mean']=.5*(predictions['original']+predictions['fresh_kumo_seed'])
    for mode,pred in predictions.items():
        score=b.scores(y,pred)
        score['fold_KS']=[b.scores(y[fold==f],pred[fold==f])['KS']for f in range(5)]
        result[mode]=score
    b.dump(dict(status='complete',metrics=result,parameters=dict(temperature=TEMPERATURE,trust=TRUST,ridge=RIDGE),
                selection_exposure='Reused development. One predeclared objective, no hyperparameter selection.'),R/'DIRECT_KS_RESULT.json')
    np.savez_compressed(R/'DIRECT_KS_OOF.npz',target=y,fold=fold,**predictions)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['run','report'])
    p.add_argument('--fold',type=int,choices=range(5))
    args=p.parse_args()
    if args.action=='run':run(args.fold)
    else:report()
