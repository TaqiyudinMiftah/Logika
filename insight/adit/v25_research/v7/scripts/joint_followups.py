"""Prespecified follow-ups after V185's validation/public disagreement."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse,hashlib,json,shutil,sys,traceback
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize,check_grad
from scipy.special import expit,logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import dump,sha,design,corrected_probability,scores
from experiments import fit_head,bootstrap
from modeling_features import error_analysis
from conservative_adapters import main as evaluate,group_objective,solve
SPECTRAL=json.loads((V/'runs/V167/RESULT.json').read_text())['columns'][18:]
assert len(SPECTRAL)==11


def local_cache(f):
    arrays=[];ids=[];checks=[]
    for part,key in [('holdout','h'),('oof','val'),('test','test')]:
        path=V/'runs/V178'/f"fold{f['fold']}_kumo_{part}.parquet"
        d=pd.read_parquet(path);assert d.application_id.tolist()==f[key].application_id.tolist()
        arrays.extend(d.prediction.to_list());ids.extend(d.application_id.to_list())
        checks.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),rows=len(d)))
    p=pd.Series(arrays,index=ids);assert p.index.is_unique and np.isfinite(p).all()
    return p,checks


def fusion_objective(w,z,d,y,penalty):
    eta=z+d@w;dev=w-1.
    return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*penalty*np.dot(dev,dev)),d.T@(expit(eta)-y)/len(y)+penalty*dev


def fit_inner_fusion(f,columns,local,spec):
    h=f['h'];y=h.target.to_numpy(int);basecols=[c for c in columns if c not in SPECTRAL]
    predictions=np.full((len(h),3),np.nan);checks=[]
    for block,(ti,qi) in enumerate(StratifiedKFold(3,shuffle=True,random_state=53027+f['fold']).split(h,y)):
        assert not np.intersect1d(ti,qi).size
        for j,(cols,k) in enumerate([(basecols,f['kh']),(columns,f['kh']),(basecols,local)]):
            head=fit_head(h.iloc[ti],k[ti],cols,.003,.5)
            predictions[qi,j]=corrected_probability(head,h.iloc[qi],k[qi])[0]
        checks.append(dict(block=block,fit_rows=len(ti),query_rows=len(qi),disjoint=True,
            fit_ids_sha256=hashlib.sha256('\n'.join(h.application_id.iloc[ti].astype(str)).encode()).hexdigest(),
            query_ids_sha256=hashlib.sha256('\n'.join(h.application_id.iloc[qi].astype(str)).encode()).hexdigest()))
    assert np.isfinite(predictions).all()
    z=logit(np.clip(predictions,1e-6,1-1e-6));d=z[:,1:]-z[:,[0]]
    with threadpool_limits(limits=2):
        fit=minimize(fusion_objective,np.ones(2),args=(z[:,0],d,y,spec['weight_penalty']),method='L-BFGS-B',jac=True,bounds=[(0.,1.5),(0.,1.5)],options=dict(maxiter=300,ftol=1e-12,gtol=1e-8))
    assert fit.success
    gradient=fusion_objective(fit.x,z[:,0],d,y,spec['weight_penalty'])[1]
    for j in range(2):
        if (fit.x[j]<1e-8 and gradient[j]>0) or (fit.x[j]>1.5-1e-8 and gradient[j]<0):gradient[j]=0.
    assert max(abs(gradient))<1e-4
    return fit.x,dict(inner_checks=checks,no_spline_used_for_weight_fit=True,no_outer_labels_in_weight_fit=True,
        weight_spectral=float(fit.x[0]),weight_local=float(fit.x[1]),objective=float(fit.fun),gradient_max_abs=float(max(abs(gradient))),iterations=int(fit.nit))


def fit_adapter(f,spec,columns):
    method=spec['method'];model=dict(kind=method,spec=spec,fold=f['fold'])
    if spec.get('fixed_score_fusion'):
        return model,dict(gradient_max_abs=0.,iterations=0,new_fitted_parameters=0),0
    local,checks=local_cache(f);model['raw_local_predictions']=local
    kh=local.reindex(f['h'].application_id).to_numpy(float)
    if method in ['joint_spectral','joint_group_ridge']:
        head=fit_head(f['h'],kh,columns,.003,.5);audit=dict(gradient_max_abs=head['gradient_max_abs'],iterations=head['iterations'])
        if method=='joint_group_ridge':
            x,z=design(head,f['h'],kh);numeric=list(head['encoder'].transformers_[0][2]);width=head['encoder'].named_transformers_['smooth'].named_steps['basis'].n_features_out_//len(numeric)
            penalty=np.full(x.shape[1],.003);penalty[0]=0.
            for j,c in enumerate(numeric):
                if c in SPECTRAL:
                    penalty[2+j*width:2+(j+1)*width]=spec['new_group_penalty'];penalty[2+len(numeric)*width+j]=spec['new_group_penalty']
            assert np.sum(penalty==spec['new_group_penalty'])==len(SPECTRAL)*(width+1)
            head['coefficients'],audit=solve(group_objective,head['coefficients'],(x,z,f['h'].target.to_numpy(int),penalty));model['penalty_vector']=penalty
        model['head']=head;count=len(head['coefficients'])
    else:
        assert method=='inner_fusion'
        w,audit=fit_inner_fusion(f,columns,kh,spec)
        basecols=[c for c in columns if c not in SPECTRAL]
        # Final fits use the full6k, after the2 mixing weights have been learned
        # exclusively from inner-held-out corrected Kumo predictions.
        model.update(weights=w,base_head=f['head'],spectral_head=fit_head(f['h'],f['kh'],columns,.003,.5),local_head=fit_head(f['h'],kh,basecols,.003,.5))
        count=sum(len(model[k]['coefficients']) for k in ['base_head','spectral_head','local_head'])+2
    audit['raw_local_cache']=checks
    return model,audit,count


def probability(model,frame,kumo,base_head):
    if model['spec'].get('fixed_score_fusion'):return corrected_probability(base_head,frame,kumo)
    local=model['raw_local_predictions'].reindex(frame.application_id).to_numpy(float);assert np.isfinite(local).all()
    if model['kind'] in ['joint_spectral','joint_group_ridge']:
        return corrected_probability(model['head'],frame,local)
    p0=corrected_probability(model['base_head'],frame,kumo)[0]
    ps=corrected_probability(model['spectral_head'],frame,kumo)[0]
    pl=corrected_probability(model['local_head'],frame,local)[0]
    z0=logit(np.clip(p0,1e-6,1-1e-6));d=np.column_stack([logit(np.clip(p,1e-6,1-1e-6))-z0 for p in [ps,pl]])
    z=z0+d@model['weights']
    return expit(z),z-logit(np.clip(kumo,1e-6,1-1e-6))


def final_prediction(model,f,p,tp):
    method=model['kind']
    if method not in ['partial_pool_composition','coverage_safeguard']:return p,tp
    results=[]
    for part,key in [('validation','val'),('test','test')]:
        needed=['V25','V185','V189'] if method=='partial_pool_composition' else ['V178','V185'];values={};hashes={}
        for source in needed:
            path=V/'runs'/source/f"fold{f['fold']}_{part}.parquet";saved=pd.read_parquet(path)
            assert saved.application_id.tolist()==f[key].application_id.tolist();values[source]=saved.prediction.to_numpy();hashes[source]=sha(path)
        if method=='partial_pool_composition':
            z=logit(np.clip(values['V185'],1e-6,1-1e-6));d=logit(np.clip(values['V189'],1e-6,1-1e-6))-logit(np.clip(values['V25'],1e-6,1-1e-6))
            pred=expit(z+.5*np.clip(d,-.1,.1))
        else:
            n=f[key].aux_allocation_count.to_numpy(float);pred=np.where(np.isfinite(n)&(n>0),values['V185'],values['V178'])
        results.append(pred);model['source_'+part+'_sha256']=hashes
    return tuple(results)


def generalization_review(out,result,tr,y,folds,reference,pred,frames):
    src=pd.read_csv(V/'runs/V185/OOF.csv',float_precision='round_trip');assert src.application_id.tolist()==tr.application_id.tolist()
    parent=src.prediction.to_numpy();interval=bootstrap(y,parent,pred,folds);b=scores(y,parent)
    wins=sum(scores(y[folds==i],pred[folds==i])['KS']>scores(y[folds==i],parent[folds==i])['KS'] for i in range(5))
    comparison=dict(control='V185',control_metrics=b,candidate_metrics=result['OOF'],delta_KS=result['OOF']['KS']-b['KS'],delta_AUC=result['OOF']['AUC']-b['AUC'],fold_wins=int(wins),confidence_interval=interval)
    result['comparison_to_V185']=comparison;dump(comparison,out/'MATCHED_V185.json')
    detail=out/'vs_V185';detail.mkdir(exist_ok=True)
    err=error_analysis(detail,tr,y,folds,parent,pred,frames)
    for extension in ['json','md']:shutil.copyfile(detail/('ERROR_ANALYSIS.'+extension),out/('ERROR_ANALYSIS_V185.'+extension))
    gap=float(np.mean([f['train_validation_head_KS_gap'] for f in result['folds']]));base=json.loads((V/'runs/V25/RESULT.json').read_text());bg=float(np.mean([f['train_validation_head_KS_gap'] for f in base['folds']]))
    flags=['REUSED_DEVELOPMENT_FOLDS','MANY_PRIOR_EXPERIMENTS','NO_INDEPENDENT_GENERALIZATION_TEST']
    if gap-bg>.005:flags.append('LARGER_TRAIN_VALIDATION_HEAD_GAP')
    if result['delta_AUC']<0:flags.append('AUC_DECREASE_VS_V25')
    if comparison['delta_AUC']<0:flags.append('AUC_DECREASE_VS_V185')
    if result['confidence_interval']['delta_KS'][0]<=0:flags.append('DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE')
    if interval['delta_KS'][0]<=0:flags.append('DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE')
    if wins<4:flags.append('FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185')
    decision='EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT'
    if comparison['delta_KS']>0 and comparison['delta_AUC']>=0 and result['delta_AUC']>=0 and wins>=4:
        decision='PROMISING_ON_DEVELOPMENT_FOLDS_NEEDS_INDEPENDENT_VALIDATION'
    review=dict(status='REVIEWED',candidate=result['experiment'],decision=decision,pooled_KS=result['OOF']['KS'],AUC=result['OOF']['AUC'],delta_AUC_vs_V25=result['delta_AUC'],delta_AUC_vs_V185=comparison['delta_AUC'],head_gap=gap,V25_head_gap=bg,flags=flags,public_incumbent=dict(id='V25',KS=.56822),development_incumbent_public_outcome=dict(id='V185',KS=.56699),causation_note='Public decline does not uniquely establish overfitting; sample variability, population drift and source instability remain possible.',selection_caveat='Intervals condition on fitted predictions and exclude search/retraining uncertainty; operational rules are heuristics, not significance tests.',top_risk_capture_vs_V185=err['risk_cutoffs'])
    result['generalization_review']=review;dump(review,out/'GENERALIZATION_REVIEW.json')
    with (out/'REPORT.md').open('a') as f:
        f.write(f"\n\nV185 comparison: deltaKS={comparison['delta_KS']:+.10f}, deltaAUC={comparison['delta_AUC']:+.10f}, wins={wins}/5, deltaKS95% CI={interval['delta_KS']}.\n\nGeneralization decision: {decision}. Flags: {', '.join(flags)}. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.\n")


def preflight():
    rng=np.random.default_rng(531);z=rng.normal(-1,1,1000);d=rng.normal(0,.15,(1000,2));y=rng.binomial(1,expit(z));w=np.array([.9,1.1])
    error=check_grad(lambda a:fusion_objective(a,z,d,y,.02)[0],lambda a:fusion_objective(a,z,d,y,.02)[1],w);assert error<1e-5
    # Artifact test uses no target labels or candidate outer-score evaluation.
    f=dict(fold=0)
    for part,key in [('holdout','h'),('oof','val'),('test','test')]:
        saved=pd.read_parquet(V/'runs/V178'/f'fold0_kumo_{part}.parquet');f[key]=saved[['application_id']]
    p,checks=local_cache(f);assert len(p)==50000
    source_results={s:json.loads((V/'runs'/s/'RESULT.json').read_text()) for s in ['V25','V167','V178','V185','V189']}
    assert source_results['V167']['columns'][18:]==SPECTRAL
    assert all(s['frozen_Kumo_fraction']==1. for s in source_results.values())
    dump(dict(status='PASS',synthetic_gradient_error=float(error),raw_cache_rows=len(p),raw_cache_checks=checks,source_sha256=sha(__file__),no_candidate_outer_scores_evaluated=True),V/'joint_followup/reports/PREFLIGHT.json')
    print('PREFLIGHT PASS',flush=True)


def main(ident):
    spec=next(s for s in json.loads((V/'joint_followup_plan.json').read_text()) if s['id']==ident)
    pre=json.loads((V/'joint_followup/reports/PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['source_sha256']==sha(__file__)
    return evaluate(ident,plan_file='joint_followup_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,method_formulas={spec['method']:spec['formula']},spectral_methods=['joint_spectral','joint_group_ridge','inner_fusion'],spectral_columns=SPECTRAL,worker_source='joint_followups.py',final_prediction=final_prediction,postprocess=generalization_review)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    try:
        if args.preflight:preflight()
        else:assert args.run;main(args.run)
    except Exception:
        if args.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json')
        raise
