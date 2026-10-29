"""Five prespecified mechanisms; no search or backbone gradient updates."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import argparse,copy,io,json,sys,traceback
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize,check_grad
from scipy.special import expit,logit
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import corrected_probability,design,dump,sha,scores,ids_hash
from experiments import fit_head
from adaptive_adapters import inner_predictions

PAIRS=[('re_price_log_posterior','re_price_posterior_sd'),('re_bureau_score_posterior','re_bureau_posterior_sd')]
VIEWS={
    'price':['re_price_log_posterior','re_price_posterior_sd'],
    'bureau':['re_bureau_score_posterior','re_bureau_posterior_sd'],
    'capacity':['re_income_geometric_mean','re_declared_income_log_gap','re_outflow_income_geometric_ratio','installment_observed_income_ratio','v2_existing_total_burden_observed'],
    'repayment':['re_late_1_posterior','re_late_30_posterior']}
MONOTONE={'re_bureau_score_posterior':-1,'re_late_30_posterior':1,'installment_observed_income_ratio':1,'v2_existing_total_burden_observed':1}
FRAME=None


def quadratic_objective(beta,x,z,y,penalty):
    eta=z+x@beta
    return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*beta@penalty@beta),x.T@(expit(eta)-y)/len(y)+penalty@beta


def optimize(x,z,y,penalty,start,bounds=None):
    with threadpool_limits(limits=2):
        r=minimize(quadratic_objective,start,args=(x,z,y,penalty),method='L-BFGS-B',jac=True,bounds=bounds,options=dict(maxiter=600,ftol=1e-12,gtol=1e-8))
    gradient=quadratic_objective(r.x,x,z,y,penalty)[1];projected=gradient.copy()
    if bounds:
        for j,(low,high) in enumerate(bounds):
            if low is not None and r.x[j]<=low+1e-8 and gradient[j]>0:projected[j]=0
            if high is not None and r.x[j]>=high-1e-8 and gradient[j]<0:projected[j]=0
    assert r.success and max(abs(projected))<1e-4,(r.message,max(abs(projected)))
    return r.x,dict(iterations=int(r.nit),gradient_max_abs=float(max(abs(projected))),objective=float(r.fun))


def integrate_uncertainty(head,frame,kumo):
    # SDs are in raw posterior-mean units, before baseline signedlog.
    nodes,weights=np.polynomial.hermite.hermgauss(3);weights=weights/np.sqrt(np.pi)
    total=np.zeros(len(frame));original=[pd.to_numeric(frame[c],errors='coerce').to_numpy(float) for c,_ in PAIRS]
    sds=[]
    for (_,s),mean in zip(PAIRS,original):
        sd=pd.to_numeric(frame[s],errors='coerce').to_numpy(float)
        assert np.all(sd[np.isfinite(sd)]>=0),'Negative posterior SD'
        sds.append(np.where(np.isfinite(sd)&np.isfinite(mean),sd,0.))
    for i,a in enumerate(nodes):
        for j,b in enumerate(nodes):
            query=frame.copy()
            for k,t in enumerate([a,b]):query[PAIRS[k][0]]=original[k]+np.sqrt(2)*sds[k]*t
            total+=weights[i]*weights[j]*corrected_probability(head,query,kumo)[0]
    return total


def function_penalty(head,x):
    penalty=.003*np.eye(x.shape[1]);penalty[0,0]=0
    names=list(head['encoder'].get_feature_names_out());groups={}
    for c in head['columns']:
        groups[c]=[j+2 for j,name in enumerate(names) if name.startswith('smooth__'+c+'_sp_') or name=='missing__missingindicator_'+c or name.startswith('category__'+c+'_')]
    assigned=[j for indices in groups.values() for j in indices]
    assert len(assigned)==len(set(assigned)) and set(assigned)==set(range(2,x.shape[1])),'Every encoded term needs one provenance block'
    for indices in groups.values():
        a=x[:,indices];a=a-a.mean(0)
        penalty[np.ix_(indices,indices)]+=.01*(a.T@a)/len(x)
    return penalty,groups


def ramp_design(frame,bundle):
    columns=[np.ones(len(frame))];masks=[]
    for c,sign in MONOTONE.items():
        value=pd.to_numeric(frame[c],errors='coerce').to_numpy(float);value=sign*np.sign(value)*np.log1p(abs(value))
        missing=~np.isfinite(value);masks.append(missing.astype(float));value=np.where(missing,bundle['medians'][c],value)
        knots=bundle['knots'][c]
        for lo,hi in zip(knots[:-1],knots[1:]):columns.append(np.clip((value-lo)/(hi-lo),0.,1.))
    return np.column_stack(columns+masks)


def fit_monotone(training):
    bundle=dict(medians={},knots={})
    for c,sign in MONOTONE.items():
        a=pd.to_numeric(training[c],errors='coerce').to_numpy(float);a=sign*np.sign(a)*np.log1p(abs(a));finite=a[np.isfinite(a)]
        bundle['medians'][c]=float(np.median(finite)) if len(finite) else 0.
        bundle['knots'][c]=np.unique(np.quantile(finite,[0,.25,.5,.75,1])) if len(finite) else np.array([0.])
    x=ramp_design(training,bundle);p=.01*np.eye(x.shape[1]);p[0,0]=0
    nramps=x.shape[1]-1-len(MONOTONE);bounds=[(None,None)]+[(0,None)]*nramps+[(None,None)]*len(MONOTONE)
    bundle['coefficients'],audit=optimize(x,np.zeros(len(training)),training.target.to_numpy(int),p,np.zeros(x.shape[1]),bounds)
    audit.update(ramp_parameters=nramps,training_rows=len(training),monotonicity_is_domain_hypothesis=True,standalone_training_metrics=scores(training.target,expit(x@bundle['coefficients'])))
    return bundle,audit


def fit_adapter(f,spec,columns):
    method=spec['method'];model=dict(method=method,head=f['head'],spec=spec);audit={};parameters=0
    if method=='posterior_quadrature':
        audit=dict(no_additional_fit=True,quadrature_points=9,posterior_independence_assumed=True,posterior_covariance_unavailable=True,means_and_SD_units_checked_in_source=True)
    elif method=='source_jackknife':
        heads=[]
        for name,removed in VIEWS.items():
            kept=[c for c in columns if c not in removed];assert len(kept)<len(columns)
            head=fit_head(f['h'],f['kh'],kept,.003,.5);heads.append(head)
        model['heads']=heads;parameters=sum(len(h['coefficients']) for h in heads)
        audit=dict(excluded_source_blocks=VIEWS,coverage_inputs_retained=True,gradient_max_abs=max(h['gradient_max_abs'] for h in heads),parameters=parameters)
    elif method=='function_space_ridge':
        head=copy.deepcopy(f['head']);x,z=design(head,f['h'],f['kh']);penalty,groups=function_penalty(head,x)
        head['coefficients'],audit=optimize(x,z,f['h'].target.to_numpy(int),penalty,head['coefficients'])
        model.update(candidate=head,penalty=penalty);parameters=len(head['coefficients']);audit.update(groups=groups,function_variance_penalty=.01,coefficient_ridge=.003,geometry_fit_rows=len(f['h']))
    elif method=='monotone_credit_expert':
        index=np.load(ROOT/'artifacts'/f"fold{f['fold']}"/'indices.npz');ti=np.r_[index['context'],index['correction']]
        assert len(np.unique(ti))==56000 and not np.intersect1d(ti,index['validation']).size
        tr=pd.read_parquet(ROOT/'data/train_features.parquet',columns=['application_id','target'])
        training=FRAME.loc[tr.application_id.iloc[ti]].copy();training['target']=tr.target.to_numpy()[ti]
        model['expert'],audit=fit_monotone(training);parameters=len(model['expert']['coefficients'])
        audit['full_outer_training_ids_sha256']=ids_hash(tr,ti)
        audit['standalone_outer_metrics']=scores(f['val'].target,expit(ramp_design(FRAME.loc[f['val'].application_id],model['expert'])@model['expert']['coefficients']))
    elif method=='inner_isotonic':
        inner,heads,assign,checks=inner_predictions(f,columns)
        iso=IsotonicRegression(y_min=1e-5,y_max=1-1e-5,out_of_bounds='clip').fit(inner,f['h'].target)
        model['isotonic']=iso;model['inner_checks']=checks;parameters=len(iso.X_thresholds_)
        audit=dict(inner_checks=checks,inner_query_targets_excluded_from_own_head=True,calibration_training_rows=6000,learned_knots=parameters,calibration_evaluated_on_outer_only=True)
    else:raise ValueError(method)
    return model,audit,parameters


def probability(model,frame,kumo,head):
    old,_=corrected_probability(model['head'],frame,kumo);kind=model['method']
    if kind=='posterior_quadrature':new=integrate_uncertainty(model['head'],frame,kumo)
    elif kind=='source_jackknife':new=np.mean([corrected_probability(h,frame,kumo)[0] for h in model['heads']],axis=0)
    elif kind=='function_space_ridge':new=corrected_probability(model['candidate'],frame,kumo)[0]
    elif kind=='inner_isotonic':new=model['isotonic'].predict(old)
    else:assert kind=='monotone_credit_expert';return corrected_probability(model['head'],frame,kumo)
    p=.95*old+.05*new
    return p,logit(np.clip(p,1e-6,1-1e-6))-logit(np.clip(kumo,1e-6,1-1e-6))


def final_prediction(model,f,p,tp):
    if model['method']!='monotone_credit_expert':return p,tp
    expert=model['expert'];pv=expit(ramp_design(FRAME.loc[f['val'].application_id],expert)@expert['coefficients']);pt=expit(ramp_design(FRAME.loc[f['test'].application_id],expert)@expert['coefficients'])
    return .95*p+.05*pv,.95*tp+.05*pt


def postprocess(out,result,tr,y,folds,reference,prediction,frames):
    from stability_worker import postprocess as common
    result['overfitting_assessment']['diagnostic']=result['spec']['rationale']+'; fixed5% new contribution, no outer tuning. Correction fits use original6k; monotone standalone expert alone uses56k. Original latent-feature construction/provenance and repeated development-selection risks remain.'
    if result['spec']['method']=='monotone_credit_expert':
        for row in result['folds']:
            row['new_expert_training_rows']=56000;row['new_expert_training_ids_sha256']=row['optimization']['full_outer_training_ids_sha256']
        result['source_cache_sha256']=sha(V/'anchored_spline/data/inputs.parquet')
    common(out,result,tr,y,folds,reference,prediction,frames)
    result['five_percent_prespecified']=True


def preflight():
    rng=np.random.default_rng(281027);n=6000;specs=json.loads((V/'conservative_mechanisms_plan.json').read_text());cols=specs[0]['input_features']
    h=pd.DataFrame({c:rng.lognormal(size=n) for c in cols if c!='employment_type'});h['employment_type']=np.resize(['salaried','self_employed','variable_income'],n)
    h['application_id']=[f'preflight_{i}' for i in range(n)];k=expit(rng.normal(-1.8,1.,n));h['target']=rng.binomial(1,k)
    h.loc[h.index[::11],'re_bureau_score_posterior']=np.nan;head=fit_head(h,k,cols,.003,.5);frame=dict(h=h,kh=k,head=head,fold=0)
    x=rng.normal(size=(100,8));z=rng.normal(size=100);y=rng.binomial(1,.2,100);pen=.01*np.eye(8);beta=rng.normal(size=8)
    err=check_grad(lambda b:quadratic_objective(b,x,z,y,pen)[0],lambda b:quadratic_objective(b,x,z,y,pen)[1],beta);assert err<1e-5
    checks=[]
    for spec in specs:
        if spec['method']=='monotone_credit_expert':
            expert,audit=fit_monotone(h);model=dict(method=spec['method'],expert=expert,head=head,spec=spec);parameters=len(expert['coefficients'])
            for name,sign in MONOTONE.items():
                q=pd.concat([h.iloc[[1]]]*80,ignore_index=True);q[name]=np.linspace(.01,10.,80)
                assert np.all(sign*np.diff(expit(ramp_design(q,expert)@expert['coefficients']))>=-1e-12)
        else:model,audit,parameters=fit_adapter(frame,spec,cols)
        q=h.iloc[:100].copy();q['employment_type']='unseen';p,_=probability(model,q,k[:100],head);q.target=1-q.target
        assert np.array_equal(p,probability(model,q,k[:100],head)[0]) and np.isfinite(p).all()
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0);assert np.array_equal(p,probability(joblib.load(stream),q,k[:100],head)[0])
        if spec['method']=='monotone_credit_expert':
            before=expit(ramp_design(q,expert)@expert['coefficients']);q.target=1-q.target;assert np.array_equal(before,expit(ramp_design(q,expert)@expert['coefficients']))
        if spec['method']=='inner_isotonic':
            grid=np.linspace(1e-5,1-1e-5,1000);assert np.all(np.diff(.95*grid+.05*model['isotonic'].predict(grid))>0)
        checks.append(dict(experiment=spec['id'],parameters=parameters,label_isolation=True,checkpoint_replay=True))
    q=h.iloc[:50].copy()
    for _,sd in PAIRS:q[sd]=0.
    assert np.max(abs(integrate_uncertainty(head,q,k[:50])-corrected_probability(head,q,k[:50])[0]))<1e-14
    dump(dict(status='PASS',synthetic_only=True,gradient_error=float(err),zero_SD_identity=True,monotone_constraints_checked=True,branches=checks,source_sha256=sha(__file__)),V/'conservative_mechanisms/reports/PREFLIGHT.json')
    print('CONSERVATIVE_MECHANISMS_PREFLIGHT_PASS',flush=True)


def main(ident):
    global FRAME
    spec=next(s for s in json.loads((V/'conservative_mechanisms_plan.json').read_text()) if s['id']==ident)
    pre=json.loads((V/'conservative_mechanisms/reports/PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['source_sha256']==sha(__file__)
    if spec['method']=='monotone_credit_expert':
        build=json.loads((V/'anchored_spline/reports/FEATURE_BUILD.json').read_text());assert build['sha256']==sha(V/'anchored_spline/data/inputs.parquet')
        FRAME=pd.read_parquet(V/'anchored_spline/data/inputs.parquet').set_index('application_id');assert FRAME.index.is_unique and 'target' not in FRAME
    from conservative_adapters import main as evaluate
    return evaluate(ident,plan_file='conservative_mechanisms_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,method_formulas={spec['method']:spec['formula']},spectral_methods=[],worker_source='conservative_mechanisms.py',final_prediction=final_prediction,postprocess=postprocess)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    try:
        if a.preflight:preflight()
        else:assert a.run;main(a.run)
    except Exception:
        if a.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/a.run/'STATUS.json')
        raise
