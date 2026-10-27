"""Resampling stability of the new high-AUC correction representation."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse,hashlib,io,json,sys,tempfile,traceback
import numpy as np
import pandas as pd
import joblib
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import dump,sha,corrected_probability,scores
from conservative_adapters import main as evaluate,SPECTRAL
from stability_worker import alternate_head,postprocess as common_review
from joint_followups import local_cache


def idhash(d):return hashlib.sha256('\n'.join(d.application_id.astype(str)).encode()).hexdigest()


def confidence_weight(variance,reference):
    variance=np.asarray(variance,float);assert np.isfinite(variance).all() and np.all(variance>=0)
    if not np.isfinite(reference) or reference<=1e-8:return np.zeros(len(variance))
    return .1*reference/(reference+variance)


def fit_adapter(f,spec,columns):
    model=dict(kind=spec['method'],spec=spec,fold=f['fold'])
    if spec.get('fixed_score_fusion'):return model,dict(gradient_max_abs=0.,iterations=0),0
    h=f['h'];y=h.target.to_numpy(int)
    if f.get('synthetic_preflight'):local=f['synthetic_local'];checks=[]
    else:local,checks=local_cache(f)
    k=local.reindex(h.application_id).to_numpy(float);heads=[];draws=[];audit=[]
    for member in range(3):
        rng=np.random.default_rng(57027+100*f['fold']+member)
        draw=np.concatenate([rng.choice(np.flatnonzero(y==c),int(np.sum(y==c)),replace=True) for c in [0,1]]);rng.shuffle(draw)
        head=alternate_head(h.iloc[draw],k[draw],columns,dict(spec,method='reduced_knots',n_knots=2));heads.append(head);draws.append(draw)
        audit.append(dict(member=member,seed=57027+100*f['fold']+member,rows=len(draw),unique_training_rows=len(np.unique(draw)),intentional_bootstrap_duplicates=True,no_outer_rows_in_draw=True))
    training=np.column_stack([corrected_probability(head,h,k)[0] for head in heads]);variance=np.var(logit(np.clip(training,1e-6,1-1e-6)),axis=1,ddof=1);reference=float(np.median(variance))
    out=Path(f['synthetic_output']) if f.get('synthetic_preflight') else V/'runs'/spec['id'];out.mkdir(parents=True,exist_ok=True)
    dump(dict(reference_median_training_logit_variance=reference,fit_partition='6k correction only',conditional_uncertainty_proxy=True,not_full_Bayesian_posterior=True,bootstrap=audit),out/f"fold{f['fold']}_resample_reference.json")
    model.update(heads=heads,draws=draws,local=local,reference_variance=reference,out=str(out),partitions={idhash(f[key]):part for key,part in [('h','training'),('val','validation'),('test','test')] if key in f})
    return model,dict(gradient_max_abs=max(h['gradient_max_abs'] for h in heads),iterations=sum(h['iterations'] for h in heads),bootstrap=audit,raw_local_cache=checks,reference_variance=reference),sum(len(h['coefficients']) for h in heads)


def probability(model,frame,k,head):
    if model['spec'].get('fixed_score_fusion'):return corrected_probability(head,frame,k)
    raw=model['local'].reindex(frame.application_id).to_numpy(float);assert np.isfinite(raw).all()
    members=np.column_stack([corrected_probability(h,frame,raw)[0] for h in model['heads']]);mean=members.mean(1);variance=np.var(logit(np.clip(members,1e-6,1-1e-6)),axis=1,ddof=1)
    part=model['partitions'].get(idhash(frame))
    if part:
        d=pd.DataFrame(dict(application_id=frame.application_id.to_numpy(),logit_variance=variance))
        for j in range(3):d[f'corrected_member{j}']=members[:,j]
        d.to_parquet(Path(model['out'])/f"fold{model['fold']}_resample_{part}.parquet",index=False)
    return mean,logit(np.clip(mean,1e-6,1-1e-6))-logit(np.clip(raw,1e-6,1-1e-6))


def final_prediction(model,f,p,tp):
    if not model['spec'].get('fixed_score_fusion'):return p,tp
    reference=json.loads((V/'runs/V219'/f"fold{f['fold']}_resample_reference.json").read_text())['reference_median_training_logit_variance'];out=[]
    for part,key in [('validation','val'),('test','test')]:
        a=pd.read_parquet(V/'runs/V193'/f"fold{f['fold']}_{part}.parquet");b=pd.read_parquet(V/'runs/V219'/f"fold{f['fold']}_{part}.parquet");v=pd.read_parquet(V/'runs/V219'/f"fold{f['fold']}_resample_{part}.parquet")
        assert a.application_id.tolist()==b.application_id.tolist()==v.application_id.tolist()==f[key].application_id.tolist()
        weight=confidence_weight(v.logit_variance.to_numpy(),reference);out.append((1-weight)*a.prediction.to_numpy()+weight*b.prediction.to_numpy())
    model.update(reference_variance=reference,zero_reference_fallback='retain V193; no finite variance reference')
    return tuple(out)


def postprocess(out,result,tr,y,folds,reference,pred,frames):
    common_review(out,result,tr,y,folds,reference,pred,frames)
    result['overfitting_assessment']['bootstrap_uncertainty_is_proxy']=True
    result['generalization_review']['head_gap_interpretation']=result['spec']['head_gap_caveat']
    dump(result['generalization_review'],out/'GENERALIZATION_REVIEW.json')
    if not result['spec'].get('fixed_score_fusion'):
        rows=[]
        for f in frames:
            d=pd.read_parquet(out/f"fold{f['fold']}_resample_validation.parquet");assert d.application_id.tolist()==f['val'].application_id.tolist()
            for j in range(3):rows.append(dict(fold=f['fold'],member=j,**scores(y[f['vi']],.75*d[f'corrected_member{j}'].to_numpy()+.25*f['sv'])))
        result['resample_heldout_members']=rows;dump(dict(members=rows,not_used_to_select_best_member=True),out/'RESAMPLE_STABILITY.json')
        dump(result['continuous_insight']|dict(resample_heldout_members=rows),out/'CONTINUOUS_INSIGHT.json')


def preflight():
    rng=np.random.default_rng(57027);n=1000;k=rng.beta(2,8,n);h=pd.DataFrame(dict(application_id=[f'SYNTHETIC_{i}' for i in range(n)],a=rng.lognormal(size=n),employment_type=np.resize(['salaried','self_employed','variable_income'],n),target=rng.binomial(1,k)))
    for c in SPECTRAL:h[c]=rng.random(n)
    spec=next(s for s in json.loads((V/'resample_plan.json').read_text()) if s['id']=='V219')
    with tempfile.TemporaryDirectory() as folder:
        f=dict(h=h,kh=k,fold=0,synthetic_preflight=True,synthetic_local=pd.Series(k,index=h.application_id),synthetic_output=folder)
        model,audit,count=fit_adapter(f,spec,['a','employment_type']+SPECTRAL);p,_=probability(model,h,k,None);assert np.isfinite(p).all();buffer=io.BytesIO();joblib.dump(model,buffer);buffer.seek(0);replay,_=probability(joblib.load(buffer),h,k,None);assert np.array_equal(p,replay)
    assert np.allclose(confidence_weight([0.,1.,9.],1.),[.1,.05,.01]);assert np.array_equal(confidence_weight([0.,1.],0.),[0.,0.])
    dump(dict(status='PASS',synthetic_only=True,bootstrap_checks=audit['bootstrap'],checkpoint_replay_exact=True,confidence_weights_verified=True,source_sha256=sha(__file__)),V/'resample/reports/PREFLIGHT.json');print('RESAMPLE_PREFLIGHT_PASS',flush=True)


def main(ident):
    spec=next(s for s in json.loads((V/'resample_plan.json').read_text()) if s['id']==ident);pre=json.loads((V/'resample/reports/PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['source_sha256']==sha(__file__)
    return evaluate(ident,plan_file='resample_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,method_formulas={spec['method']:spec['formula']},spectral_methods=[] if spec.get('fixed_score_fusion') else [spec['method']],spectral_columns=SPECTRAL,worker_source='resample_worker.py',final_prediction=final_prediction,postprocess=postprocess)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    try:
        if args.preflight:preflight()
        else:assert args.run;main(args.run)
    except Exception:
        if args.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json')
        raise
