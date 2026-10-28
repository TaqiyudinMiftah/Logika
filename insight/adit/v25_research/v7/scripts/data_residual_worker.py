"""Isolate incremental item information while preserving the fitted V25 head."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
from pathlib import Path
import argparse
import io
import json
import shutil
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / 'v7'
sys.path[:0] = [str(ROOT / 'scripts'), str(V / 'scripts')]
from baseline import design, dump, sha, scores
from experiments import bootstrap, fit_head
from data_fe_worker import signed, extra_basis, solve


def fit_adapter(frame, spec, columns):
    h = frame['h']; names = spec['input_features']
    raw = signed(h, names); count = np.isfinite(raw).sum(0)
    mean = np.divide(np.nansum(raw, axis=0), count, out=np.zeros(len(names)), where=count > 0)
    variance = np.divide(np.nansum((raw-mean)**2, axis=0), count,
                         out=np.zeros(len(names)), where=count > 0)
    scale = np.sqrt(variance); scale[scale < 1e-8] = 1.
    model = dict(head=frame['head'], columns=names, means=mean, scales=scale,
                 strength=spec['strength'], project=spec['project'])
    x, z = design(frame['head'], h, frame['kh'])
    u = extra_basis(h, model)
    projection = np.linalg.lstsq(x, u, rcond=1e-6)[0] if spec['project'] else np.zeros((x.shape[1], len(names)))
    residual = u - x @ projection
    eta = z + .5 * (x @ frame['head']['coefficients'])
    beta, audit = solve(np.zeros(len(names)), residual, eta, h.target.to_numpy(int), np.full(len(names), .01))
    model.update(projection=projection, coefficients=beta)
    audit.update(base_coefficients_preserved_exactly=True, training_only_projection=spec['project'],
                 projection_rcond=1e-6 if spec['project'] else None,
                 residual_sd=residual.std(0).tolist(), input_names=names,
                 residual_base_cross_moment_max=float(np.max(abs(x.T @ residual / len(h)))),
                 residual_rescaled_after_projection=False,
                 undefined_values_neutral_before_projection=True)
    return model, audit, len(beta)


def probability(model, frame, kumo, head):
    x, z = design(model['head'], frame, kumo)
    residual = extra_basis(frame, model) - x @ model['projection']
    delta = .5 * (x @ model['head']['coefficients']) + model['strength'] * (residual @ model['coefficients'])
    return expit(z + delta), delta


def postprocess(out, result, tr, y, folds, reference, prediction, frames):
    from stability_worker import postprocess as common
    common(out, result, tr, y, folds, reference, prediction, frames)
    for source in [V/'scripts/data_fe_worker.py', V/'scripts/item_price_resilience_features.py',
                   V/'data_fe/reports/FEATURE_BUILD.json', V/'data_fe/reports/FEATURE_CATALOG.csv']:
        shutil.copyfile(source, out/('SOURCE_' + source.name))
    result['raw_item_feature_build'] = json.loads((V/'data_fe/reports/FEATURE_BUILD.json').read_text())
    result['overfitting_assessment']['diagnostic'] += '; V25 teacher was fitted on the same correction rows. Projection removes linear overlap in the encoded training basis only; it does not prove statistical independence or create an independent validation set.'
    if result['experiment'] != 'V251':
        d = pd.read_csv(V/'runs/V251/OOF.csv', float_precision='round_trip')
        assert d.application_id.tolist() == tr.application_id.tolist()
        control = d.prediction.to_numpy(); metrics = scores(y, control)
        interval = bootstrap(y, control, prediction, folds)
        comparison = dict(control='V251', control_metrics=metrics, candidate_metrics=result['OOF'],
            delta_KS=result['OOF']['KS']-metrics['KS'], delta_AUC=result['OOF']['AUC']-metrics['AUC'],
            fold_wins=sum(scores(y[folds==i],prediction[folds==i])['KS'] > scores(y[folds==i],control[folds==i])['KS'] for i in range(5)),
            confidence_interval=interval,
            caveat='V252 isolates added values with the same projection; V253 also changes projection and cannot attribute its difference to values alone.')
        result['comparison_to_support_control'] = comparison
        dump(comparison, out/'MATCHED_SUPPORT_CONTROL.json')
    if result['experiment'] == 'V253':
        d = pd.read_csv(V/'runs/V252/OOF.csv', float_precision='round_trip')
        assert d.application_id.tolist() == tr.application_id.tolist()
        control = d.prediction.to_numpy(); metrics = scores(y, control)
        comparison = dict(control='V252', control_metrics=metrics, candidate_metrics=result['OOF'],
            delta_KS=result['OOF']['KS']-metrics['KS'], delta_AUC=result['OOF']['AUC']-metrics['AUC'],
            fold_wins=sum(scores(y[folds==i],prediction[folds==i])['KS'] > scores(y[folds==i],control[folds==i])['KS'] for i in range(5)),
            confidence_interval=bootstrap(y, control, prediction, folds),
            interpretation='Matched16 inputs and fixed V25 head; this contrast removes the training-only projection.')
        result['comparison_to_projection_control'] = comparison
        dump(comparison, out/'MATCHED_PROJECTION_CONTROL.json')


def preflight():
    rng = np.random.default_rng(27841); n = 1100
    k = expit(rng.normal(-1.8, 1., n))
    h = pd.DataFrame(dict(a=rng.normal(size=n), employment_type=np.resize(['salaried','self_employed'],n), target=rng.binomial(1,k)))
    specs = json.loads((V/'data_residual_plan.json').read_text())
    names = list(dict.fromkeys(c for s in specs for c in s['input_features']))
    for c in names:
        h[c] = rng.normal(size=n)
    h.loc[h.index[::5],names[-1]] = np.nan
    head = fit_head(h,k,['a','employment_type'],.003,.5)
    original = head['coefficients'].copy()
    frame = dict(h=h,kh=k,head=head)
    branches=[]
    for spec in specs:
        model,audit,count = fit_adapter(frame,spec,[])
        assert np.array_equal(head['coefficients'],original)
        query=h.iloc[:100].copy(); query.employment_type='unseen'; query['target']=1-query.target
        before=joblib.hash(model)
        p,_=probability(model,query,k[:100],head)
        query['target']=1-query.target
        label_check,_=probability(model,query,k[:100],head)
        assert np.array_equal(p,label_check)
        assert np.isfinite(p).all() and before==joblib.hash(model)
        if spec['project']:
            assert audit['residual_base_cross_moment_max'] < 1e-7
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
        replay,_=probability(joblib.load(stream),query,k[:100],head)
        assert np.array_equal(p,replay)
        branches.append(dict(experiment=spec['id'],added_parameters=count,projection_overlap=audit['residual_base_cross_moment_max'],gradient=audit['gradient_max_abs']))
    dump(dict(status='PASS',synthetic_only=True,base_head_unchanged=True,query_does_not_refit=True,
              checkpoint_replay_exact=True,branches=branches,source_sha256=sha(__file__)),V/'data_residual/reports/PREFLIGHT.json')
    print('DATA_RESIDUAL_PREFLIGHT_PASS',flush=True)


def main(ident):
    from conservative_adapters import main as evaluate
    spec=next(s for s in json.loads((V/'data_residual_plan.json').read_text()) if s['id']==ident)
    receipt=json.loads((V/'data_residual/reports/PREFLIGHT.json').read_text())
    build=json.loads((V/'data_fe/reports/FEATURE_BUILD.json').read_text())
    assert receipt['status']=='PASS' and receipt['source_sha256']==sha(__file__)
    assert build['status']=='COMPLETE' and not build['target_used']
    assert build['feature_sha256']==sha(V/'data_fe/data/item_price_resilience.parquet')
    return evaluate(ident,plan_file='data_residual_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,
        method_formulas={spec['method']:spec['formula']},spectral_methods=[spec['method']],spectral_columns=spec['input_features'],
        worker_source='data_residual_worker.py',postprocess=postprocess,feature_cache_file='v7/data_fe/data/item_price_resilience.parquet')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:
        if args.preflight:preflight()
        else:assert args.run;main(args.run)
    except Exception:
        if args.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json')
        raise
