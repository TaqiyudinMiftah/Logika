"""Low-capacity, matched tests of new item/price/capacity information."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse
import io
import json
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import design,corrected_probability,dump,sha,scores
from experiments import fit_head,bootstrap


def objective(beta,x,z,y,penalty):
    eta=z+x@beta
    return float(np.mean(np.logaddexp(0.,eta)-y*eta)+.5*np.dot(penalty,beta*beta)),x.T@(expit(eta)-y)/len(y)+penalty*beta


def solve(beta,x,z,y,penalty):
    beta=beta.copy()
    with threadpool_limits(limits=2):
        for iteration in range(60):
            value,g=objective(beta,x,z,y,penalty)
            if np.max(abs(g))<1e-7:break
            p=expit(z+x@beta);hessian=(x.T*(p*(1-p)))@x/len(y)+np.diag(penalty)
            step=np.linalg.solve(hessian,g);descent=float(g@step);assert descent>0.
            length=1.
            for _ in range(40):
                candidate=beta-length*step;new,_=objective(candidate,x,z,y,penalty)
                if np.isfinite(new) and new<=value-1e-4*length*descent:break
                length*=.5
            else:raise RuntimeError('Feature head line search did not converge')
            beta=candidate
        value,g=objective(beta,x,z,y,penalty)
    assert np.max(abs(g))<1e-6
    return beta,dict(objective=value,gradient_max_abs=float(np.max(abs(g))),iterations=iteration+1,solver='Analytic Newton with Armijo backtracking')


def signed(frame,columns):
    a=frame[columns].to_numpy(float,copy=True);a[~np.isfinite(a)]=np.nan
    return np.sign(a)*np.log1p(abs(a))


def extra_basis(frame,model):
    a=signed(frame,model['columns']);ok=np.isfinite(a)
    # Undefined raw ratios remain NaN in feature cache. In this standardized
    # model basis, unavailable evidence contributes zero; common availability
    # indicators are present in EVERY experiment, including the support control.
    return np.where(ok,(a-model['means'])/model['scales'],0.)


def fit_adapter(f,spec,columns):
    added=spec['input_features'];h=f['h'];head=f['head'];a=signed(h,added);ok=np.isfinite(a);n=ok.sum(0)
    means=np.divide(np.nansum(a,axis=0),n,out=np.zeros(len(added)),where=n>0)
    ss=np.nansum((a-means)**2,axis=0);variance=np.divide(ss,n,out=np.zeros(len(added)),where=n>0);scales=np.sqrt(variance);scales[scales<1e-8]=1.
    model=dict(head=head,columns=added,means=means,scales=scales)
    base,z=design(head,h,f['kh']);x=np.column_stack([base,extra_basis(h,model)])
    penalty=np.r_[np.full(base.shape[1],.003),np.full(len(added),spec['new_penalty'])];penalty[0]=0.
    beta,audit=solve(np.r_[head['coefficients'],np.zeros(len(added))],x,z,h.target.to_numpy(int),penalty)
    model.update(coefficients=beta,penalty=penalty)
    audit.update(base_coefficients=base.shape[1],new_linear_coefficients=len(added),extra_missing_indicator_columns=0,
                 availability_features_shared_across_batch=True,all_missing_training_features=[c for c,v in zip(added,n) if v==0],
                 input_feature_names=added,training_only_scaling=True)
    return model,audit,len(beta)


def probability(model,frame,k,head):
    base,z=design(model['head'],frame,k);x=np.column_stack([base,extra_basis(frame,model)]);delta=x@model['coefficients']
    return expit(z+.5*delta),delta


def postprocess(out,result,tr,y,folds,reference,pred,frames):
    import shutil
    for source in [V/'scripts/item_price_resilience_features.py',V/'data_fe/reports/FEATURE_BUILD.json',V/'data_fe/reports/FEATURE_CATALOG.csv']:
        shutil.copyfile(source,out/('SOURCE_'+source.name))
    from stability_worker import postprocess as common
    common(out,result,tr,y,folds,reference,pred,frames)
    result['raw_item_feature_build']=json.loads((V/'data_fe/reports/FEATURE_BUILD.json').read_text())
    result['overfitting_assessment']['diagnostic']+='; repeated feature search remains; finite-value contribution and source availability are separated with a shared support control and no repeated missing indicators for new values.'
    if result['experiment']!='V243':
        d=pd.read_csv(V/'runs/V243/OOF.csv',float_precision='round_trip');assert d.application_id.tolist()==tr.application_id.tolist()
        control=d.prediction.to_numpy();m=scores(y,control);ci=bootstrap(y,control,pred,folds)
        comparison=dict(control='V243',control_metrics=m,candidate_metrics=result['OOF'],delta_KS=result['OOF']['KS']-m['KS'],delta_AUC=result['OOF']['AUC']-m['AUC'],
                        fold_wins=sum(scores(y[folds==i],pred[folds==i])['KS']>scores(y[folds==i],control[folds==i])['KS'] for i in range(5)),confidence_interval=ci)
        result['comparison_to_support_control']=comparison;dump(comparison,out/'MATCHED_SUPPORT_CONTROL.json')
        with (out/'REPORT.md').open('a',encoding='utf-8') as f:f.write(f"\nUnique information vs shared support controlV243: deltaKS={comparison['delta_KS']:+.10f}, deltaAUC={comparison['delta_AUC']:+.10f}, paired deltaKS CI={ci['delta_KS']}. No independent holdout is created by this comparison.\n")


def preflight():
    rng=np.random.default_rng(66027);n=1200;k=expit(rng.normal(-1.8,1.,n));h=pd.DataFrame(dict(a=rng.lognormal(size=n),employment_type=np.resize(['salaried','self_employed','variable_income'],n),target=rng.binomial(1,k)))
    specs=json.loads((V/'data_fe_plan.json').read_text());all_extra=list(dict.fromkeys(c for s in specs for c in s['input_features']))
    for c in all_extra:h[c]=rng.normal(size=n)
    value_names=[c for c in all_extra if c not in specs[0]['input_features']]
    for c in value_names:h.loc[h.index[::4],c]=np.nan
    base=fit_head(h,k,['a','employment_type'],.003,.5);f=dict(h=h,kh=k,head=base)
    x=rng.normal(size=(100,5));x[:,0]=1.;z=rng.normal(size=100);y=rng.binomial(1,expit(z));pen=np.full(5,.01);pen[0]=0.;b=rng.normal(0,.1,5)
    error=check_grad(lambda q:objective(q,x,z,y,pen)[0],lambda q:objective(q,x,z,y,pen)[1],b);assert error<1e-5
    branches=[]
    for s in specs:
        model,audit,count=fit_adapter(f,s,[]);q=h.iloc[:120].copy();q.employment_type='UNSEEN';p,_=probability(model,q,k[:120],base)
        assert np.isfinite(p).all() and np.all((p>0)&(p<1))
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0);p2,_=probability(joblib.load(stream),q,k[:120],base);assert np.array_equal(p,p2)
        e=extra_basis(h,model)
        for j,c in enumerate(s['input_features']):assert np.all(e[h[c].isna().to_numpy(),j]==0.)
        branches.append(dict(experiment=s['id'],parameters=count,checkpoint_replay_exact=True,gradient_max_abs=audit['gradient_max_abs']))
    dump(dict(status='PASS',synthetic_only=True,gradient_error=float(error),undefined_extra_values_neutral_with_explicit_shared_availability=True,branches=branches,source_sha256=sha(__file__)),V/'data_fe/reports/PREFLIGHT.json')
    print('DATA_FEATURE_MODEL_PREFLIGHT_PASS',flush=True)


def main(ident):
    from conservative_adapters import main as evaluate
    spec=next(s for s in json.loads((V/'data_fe_plan.json').read_text()) if s['id']==ident)
    pre=json.loads((V/'data_fe/reports/PREFLIGHT.json').read_text());build=json.loads((V/'data_fe/reports/FEATURE_BUILD.json').read_text())
    assert pre['status']=='PASS' and pre['source_sha256']==sha(__file__)
    assert build['status']=='COMPLETE' and not build['target_used']
    assert build['feature_sha256']==sha(V/'data_fe/data/item_price_resilience.parquet')
    return evaluate(ident,plan_file='data_fe_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,
                    method_formulas={spec['method']:spec['formula']},spectral_methods=[spec['method']],spectral_columns=spec['input_features'],
                    worker_source='data_fe_worker.py',postprocess=postprocess,feature_cache_file='v7/data_fe/data/item_price_resilience.parquet')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    try:
        if a.preflight:preflight()
        else:assert a.run;main(a.run)
    except Exception:
        if a.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/a.run/'STATUS.json')
        raise
