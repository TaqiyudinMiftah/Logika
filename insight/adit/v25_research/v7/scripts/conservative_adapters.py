"""Five prespecified, small corrections/ensembles on frozen V25 predictions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,gc,json,shutil,sys,time,traceback,urllib.request
from platform_runtime import fcntl,resource
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import SplineTransformer,StandardScaler,RobustScaler
from sklearn.kernel_approximation import Nystroem
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import dump,sha,scores,ids_hash,design,corrected_probability
from experiments import fit_head,bootstrap
from modeling_features import load,error_analysis
SPECTRAL=['beh_spectral_spend_power_7d','beh_spectral_spend_power_14d','beh_spectral_spend_power_30d','beh_spectral_spend_periodicity_strength','beh_spectral_orders_power_7d','beh_spectral_orders_power_14d','beh_spectral_orders_power_30d','beh_spectral_orders_periodicity_strength','beh_spectral_window_supported']
KERNEL=['re_price_log_posterior','re_bureau_score_posterior','re_late_30_posterior','installment_observed_income_ratio','v2_existing_total_burden_observed','risk_conditional_negative_months','risk_conditional_nonnegative_months','aux_account_months']

def residual_objective(beta,x,offset,y,penalty,tether):
    delta=x@beta;eta=offset+delta
    value=float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*penalty*(beta@beta)+.5*tether*np.mean(delta*delta))
    grad=x.T@(expit(eta)-y+tether*delta)/len(y)+penalty*beta
    return value,grad

def group_objective(beta,x,offset,y,penalty):
    eta=offset+x@beta
    return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.dot(penalty,beta*beta)),x.T@(expit(eta)-y)/len(y)+penalty*beta

def solve(fun,initial,args):
    with threadpool_limits(limits=2):fit=minimize(fun,initial,args=args,jac=True,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-11,gtol=1e-7))
    value,gradient=fun(fit.x,*args);assert fit.success and np.max(abs(gradient))<1e-4,(fit.message,np.max(abs(gradient)))
    return fit.x,dict(objective=value,gradient_max_abs=float(np.max(abs(gradient))),iterations=int(fit.nit))

def signed(frame,cols):
    a=frame[cols].to_numpy(dtype=float,copy=True);a[~np.isfinite(a)]=np.nan
    return np.sign(a)*np.log1p(abs(a))

def fit_adapter(f,spec,cols):
    method=spec['method'];h=f['h'];y=h.target.to_numpy(int)
    teacher,_=corrected_probability(f['head'],h,f['kh']);offset=logit(np.clip(teacher,1e-6,1-1e-6))
    if method=='group_ridge':
        head=fit_head(h,f['kh'],cols,.003,.5);x,z=design(head,h,f['kh']);penalty=np.full(x.shape[1],.003);penalty[0]=0
        numeric=list(head['encoder'].transformers_[0][2]);n=len(numeric);width=head['encoder'].named_transformers_['smooth'].named_steps['basis'].n_features_out_//n
        for j,c in enumerate(numeric):
            if c in SPECTRAL:
                penalty[2+j*width:2+(j+1)*width]=spec['new_group_penalty'];penalty[2+n*width+j]=spec['new_group_penalty']
        assert np.sum(penalty==spec['new_group_penalty'])==len(SPECTRAL)*(width+1)
        head['coefficients'],audit=solve(group_objective,head['coefficients'],(x,z,y,penalty));head.update(audit)
        return dict(kind=method,head=head,penalty_vector=penalty,columns=cols),audit,len(head['coefficients'])
    if method=='source_average':return dict(kind=method,sources=spec['sources'],weights=spec['weights']),dict(gradient_max_abs=0.,iterations=0),0
    if method=='tree_residual':
        from modeling_experts import matrix
        from lightgbm import LGBMClassifier
        x,reference,cats=matrix(h,cols)
        config=dict(n_estimators=80,max_depth=2,num_leaves=4,min_child_samples=200,learning_rate=.02,reg_lambda=50.,reg_alpha=1.,min_split_gain=.05,max_bin=31,n_jobs=2,verbosity=-1,boost_from_average=False,random_state=45027+f['fold'])
        model=LGBMClassifier(**config)
        with threadpool_limits(limits=2):model.fit(x,y,init_score=offset,categorical_feature=cats)
        delta=model.predict(x,raw_score=True)
        loss=lambda z:float(np.mean(np.logaddexp(0,z)-y*z))
        assert loss(offset+delta)<=loss(offset)+1e-5,'Residual model must add its margin to the frozen teacher offset'
        audit=dict(gradient_max_abs=0.,iterations=80,training_logloss_before=loss(offset),training_logloss_after_full_residual=loss(offset+delta),fixed_iterations=True,no_outer_early_stopping=True)
        return dict(kind=method,model=model,reference=reference,columns=cols,strength=spec['strength'],configuration=config),audit,320
    if method=='reliability_residual':
        encoder=Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('spline',SplineTransformer(n_knots=3,degree=2,knots='quantile',extrapolation='constant')),('scale',StandardScaler())])
        a=signed(h,SPECTRAL);basis=encoder.fit_transform(a);missing=MissingIndicator(features='all',sparse=False,error_on_new=False).fit(a)
        x=np.column_stack([np.ones(len(h)),basis,missing.transform(a)])
        support=(h.beh_spectral_window_supported.fillna(0).to_numpy(float)>0);n=h.aux_orders90.to_numpy(float);w=np.where(support&np.isfinite(n)&(n>0),n/(n+10.),0.);x*=w[:,None]
        model=dict(kind=method,encoder=encoder,missing=missing,columns=SPECTRAL,strength=spec['strength'])
    else:
        assert method=='kernel_residual'
        a=signed(h,KERNEL);fill=SimpleImputer(strategy='median',keep_empty_features=True);filled=fill.fit_transform(a);scale=RobustScaler().fit(filled)
        x0=np.column_stack([np.clip(scale.transform(filled),-5,5),np.isnan(a).astype(float)])
        kernel=Nystroem(kernel='rbf',gamma=1/x0.shape[1],n_components=spec['components'],random_state=48027+f['fold']);x=kernel.fit_transform(x0)
        model=dict(kind=method,fill=fill,scale=scale,kernel=kernel,columns=KERNEL,strength=spec['strength'])
    beta,audit=solve(residual_objective,np.zeros(x.shape[1]),(x,offset,y,spec['penalty'],spec['teacher_tether']));model['coefficients']=beta
    return model,audit,len(beta)

def probability(model,d,kumo,base_head):
    if model['kind']=='group_ridge':return corrected_probability(model['head'],d,kumo)
    p,base_delta=corrected_probability(base_head,d,kumo);z=logit(np.clip(p,1e-6,1-1e-6));kind=model['kind']
    if kind=='source_average':return p,base_delta
    if kind=='tree_residual':
        from modeling_experts import matrix
        x,_,_=matrix(d,model['columns'],model['reference']);delta=np.clip(model['model'].predict(x,raw_score=True),-.5,.5)
    elif kind=='reliability_residual':
        a=signed(d,model['columns']);x=np.column_stack([np.ones(len(d)),model['encoder'].transform(a),model['missing'].transform(a)])
        n=d.aux_orders90.to_numpy(float);support=d.beh_spectral_window_supported.fillna(0).to_numpy(float)>0;w=np.where(support&np.isfinite(n)&(n>0),n/(n+10.),0.);delta=(x@model['coefficients'])*w
    else:
        a=signed(d,model['columns']);x0=np.column_stack([np.clip(model['scale'].transform(model['fill'].transform(a)),-5,5),np.isnan(a).astype(float)]);delta=model['kernel'].transform(x0)@model['coefficients']
    result=expit(z+model['strength']*delta)
    if kind=='reliability_residual' and np.any(w==0):
        result[w==0]=p[w==0];assert np.array_equal(result[w==0],p[w==0])
    return result,model['strength']*delta

def main(ident, plan_file="conservative_plan.json", adapter_fit=None, adapter_probability=None, method_formulas=None, spectral_methods=None, spectral_columns=None, worker_source="adaptive_adapters.py", final_prediction=None, postprocess=None, feature_cache_file=None):
    fit_fn=adapter_fit or fit_adapter; predict_fn=adapter_probability or probability
    spectral_inputs=spectral_columns if spectral_columns is not None else SPECTRAL
    started=time.time();out=V/'runs'/ident;out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists()
    lock=(out/'adapter.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=next(s for s in json.loads((V/plan_file).read_text()) if s['id']==ident)
    gate=json.loads((V/'modeling/reports/REPO_REVIEW_GATE.json').read_text());assert gate['experiment']==ident
    req=urllib.request.Request('https://api.github.com/repos/TaqiyudinMiftah/Logika/git/ref/heads/main?fresh='+str(time.time_ns()),headers={'User-Agent':'Logika-research','Cache-Control':'no-cache'})
    with urllib.request.urlopen(req,timeout=20) as response:head=json.load(response)
    assert head['object']['sha']==gate['reviewed_head'],'Review new peer work before fitting'
    dump(dict(status='LOADING_FROZEN_V25',experiment=ident,pid=os.getpid()),out/'STATUS.json')
    for file in [Path(__file__),ROOT/'scripts/baseline.py',V/'scripts/experiments.py',V/'scripts/modeling_features.py',V/'scripts/platform_runtime.py']:shutil.copyfile(file,out/('SOURCE_'+file.name))
    if adapter_fit is not None:
        shutil.copyfile(V/'scripts'/worker_source,out/('SOURCE_'+worker_source))
    loaded=load();tr,te,sample,y,folds,reference,frames,parent=loaded;method=spec['method'];use_spectral=method in (spectral_methods if spectral_methods is not None else ['group_ridge','reliability_residual','tree_residual']);cols=parent['columns']+(spectral_inputs if use_spectral else [])
    cache_path=(ROOT/feature_cache_file) if feature_cache_file is not None else V/'light_fe/data/spectral_features.parquet'
    cache=pd.read_parquet(cache_path).set_index('application_id') if use_spectral else None
    oof=np.full(len(tr),np.nan);pt=np.zeros(len(sample));checks=[];extra_bins={'orders90_exposure_band':np.full(len(tr),'missing',dtype=object)}
    for original in frames:
        f=dict(original)
        if use_spectral:
            for part in ['h','val','test']:
                d=f[part].copy();assert d.application_id.isin(cache.index).all();d[spectral_inputs]=cache.reindex(d.application_id)[spectral_inputs].to_numpy();f[part]=d
        model,audit,trainable=fit_fn(f,spec,cols)
        n90=f['val'].aux_orders90.to_numpy(float)
        extra_bins['orders90_exposure_band'][f['vi']]=np.select([~np.isfinite(n90),n90<=0,n90<=3,n90<=10],['missing','zero','1_to3','4_to10'],default='over10')
        ph,dh=predict_fn(model,f['h'],f['kh'],f['head']);pv,dv=predict_fn(model,f['val'],f['kv'],f['head']);ptest,dt=predict_fn(model,f['test'],f['kt'],f['head'])
        p=.75*pv+.25*f['sv'];tp=.75*ptest+.25*f['st']
        if method=='source_average':
            p=np.zeros(len(f['val']));tp=np.zeros(len(f['test']))
            for source,w in zip(spec['sources'],spec['weights']):
                sr=json.loads((V/'runs'/source/'RESULT.json').read_text());assert sr['frozen_Kumo_fraction']==1. and sr['all_outer_indices_identical']
                for part,d,pred in [('validation',f['val'],p),('test',f['test'],tp)]:
                    saved=pd.read_parquet(V/'runs'/source/f"fold{f['fold']}_{part}.parquet");assert saved.application_id.tolist()==d.application_id.tolist();pred+=w*saved.prediction.to_numpy()
        if final_prediction is not None:p,tp=final_prediction(model,f,p,tp)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.isfinite(tp).all() and np.all((tp>=0)&(tp<=1))
        oof[f['vi']]=p;pt+=pd.Series(tp,index=f['test'].application_id).reindex(sample.application_id).to_numpy()/5
        m,b=scores(y[f['vi']],p),scores(y[f['vi']],reference[f['vi']]);train=scores(f['h'].target,ph);corrected=scores(y[f['vi']],pv)
        row=dict(fold=f['fold'],metrics=m,baseline=b,delta_KS=m['KS']-b['KS'],delta_AUC=m['AUC']-b['AUC'],parameters=trainable+(0 if spec.get('replaces_base_head') or method in ['group_ridge','delete_block_bagging','group_regret_robust_head'] else len(f['head']['coefficients'])),new_trainable_parameters=trainable,correction_rows=6000,head_training_KS=train['KS'],heldout_corrected_Kumo_KS=corrected['KS'],train_validation_head_KS_gap=train['KS']-corrected['KS'],train_ids_sha256=ids_hash(tr,f['hi']),validation_ids_sha256=ids_hash(tr,f['vi']),optimization=audit)
        if method=='source_average' or spec.get('fixed_score_fusion'):row['head_gap_caveat']='Unchanged V25 head diagnostic; fixed blend has no trained meta-parameters'
        if spec.get('head_gap_caveat'):row['head_gap_caveat']=spec['head_gap_caveat']
        checks.append(row);joblib.dump(dict(adapter=model,base_head=f['head'],spec=spec,kumo_weight=.75,original_spline=True),out/f"fold{f['fold']}_correction.joblib",compress=3)
        pd.DataFrame(dict(application_id=f['val'].application_id,prediction=p,correction_logit=dv)).to_parquet(out/f"fold{f['fold']}_validation.parquet",index=False)
        pd.DataFrame(dict(application_id=f['test'].application_id,prediction=tp)).to_parquet(out/f"fold{f['fold']}_test.parquet",index=False)
        dump(dict(status='FOLD_COMPLETE',experiment=ident,pid=os.getpid(),fold=f['fold'],metrics=m),out/'STATUS.json');print('FOLD',ident,f['fold'],json.dumps(m),flush=True)
        del model;gc.collect()
    assert np.isfinite(oof).all() and np.isfinite(pt).all();m,b=scores(y,oof),scores(y,reference);interval=bootstrap(y,reference,oof,folds)
    submission=V/'submissions'/f'submission_{ident}_V25_conservative.csv';pd.DataFrame(dict(application_id=sample.application_id,default_probability=pt)).to_csv(submission,index=False,lineterminator='\n')
    pd.DataFrame(dict(application_id=tr.application_id,target=y,fold=folds,prediction=oof,baseline_prediction=reference)).to_csv(out/'OOF.csv',index=False,lineterminator='\n');error_analysis(out,tr,y,folds,reference,oof,frames,extra_bins)
    formulas=dict(group_ridge='V25 offset design plus9 spectral inputs; meanBCE+.003/2||base_beta||ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²+.03/2||new_beta||ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²; inference strength.5',reliability_residual='eta=eta_V25+.25*w*phi_spectral@gamma; w=I(spectral_supported)*n_completed90/(n_completed90+10); fitBCE+.03/2||gamma||ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²+.2/2 mean((w*phi@gamma)^2)',source_average='p_final=.5*p_V167+.5*p_V146; fixed matched-fold score average',kernel_residual='eta=eta_V25+.25*Nystrom64_RBF(robust_scaled8inputs+missing)@gamma; fitBCE+.05/2||gamma||ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²+.2/2 mean(deltaÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²)',tree_residual='eta=eta_V25+.15*clip(GBDT80_depth2_residual,-.5,.5); tree training init_score=eta_V25; fixed trees, no outer early stopping')
    formulas.update(method_formulas or {})
    formula=formulas[method]+(' ; final=.75*sigmoid(eta)+.25*original_spline77' if method_formulas is None and method!='source_average' else '')+'; mean5fold test probabilities'
    result=dict(status='COMPLETE',experiment=ident,spec=spec,columns=cols,new_features=parent['new_features']+(spectral_inputs if use_spectral else []),support_features=parent['support_features'],context_columns=json.loads((ROOT/'BASELINE_LOCK.json').read_text())['Kumo']['features'],OOF=m,baseline=b,comparison_baseline='V25',delta_KS=m['KS']-b['KS'],delta_AUC=m['AUC']-b['AUC'],mean_fold_KS=float(np.mean([q['metrics']['KS'] for q in checks])),std_fold_KS=float(np.std([q['metrics']['KS'] for q in checks],ddof=1)),delta_mean_fold_KS=float(np.mean([q['delta_KS'] for q in checks])),fold_wins=sum(q['delta_KS']>0 for q in checks),folds=checks,confidence_interval=interval,modeling_formula=formula,modeling_objective=formulas[method],controller_source='v7/scripts/conservative_adapters.py',source_sha256=sha(__file__),frozen_Kumo_fraction=1.,backbone_retrained=False,backbone_gradient_updated=False,in_context_refitted=False,original_spline_unchanged=True,all_outer_indices_identical=True,point_in_time_certified=False,inherited_baseline_snapshot_uncertainty=True,features_changed_vs_V25=use_spectral,submission=str(submission.relative_to(ROOT)),submission_sha256=sha(submission),public_score=None,elapsed_seconds=time.time()-started,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,overfitting_assessment=dict(development_folds_reused=True,independent_holdout=False,hyperparameters_prespecified=True,selection_corrected_CI=False,public_score_unknown=True,risk='HIGH_SEARCH_MULTIPLICITY',diagnostic='Residual teacher was already fitted on the6k correction rows; training gains are optimistic. All new basis/reference fitting uses correction rows only, all outer labels held out.'),error_analysis=dict(json=str((out/'ERROR_ANALYSIS.json').relative_to(ROOT)),markdown=str((out/'ERROR_ANALYSIS.md').relative_to(ROOT)),cases=str((out/'ERROR_CASES.csv').relative_to(ROOT))))
    result['submission']=submission.relative_to(ROOT).as_posix()
    result['execution_host']='local_windows' if os.name=='nt' else 'node480'
    result['source_data_host']='node480'
    if os.name=='nt':
        import importlib.metadata
        result['packages']={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','pyarrow','joblib']}
    if use_spectral:
        if feature_cache_file is None:result['spectral_cache_sha256']=sha(cache_path)
        else:result['engineered_feature_cache']=dict(file=cache_path.relative_to(ROOT).as_posix(),sha256=sha(cache_path))
    if adapter_fit is not None:
        result['additional_source_sha256']=sha(V/'scripts'/worker_source);result['controller_source']='v7/scripts/'+worker_source
        result['overfitting_assessment']['diagnostic']=spec['rationale']+'; all new fitting uses correction rows only; repeated development selection remains.'
        result['overfitting_assessment']['inner_crossfit_used']=method in ['crossfit_calibration','delete_block_bagging','partial_pool_group_calibration','inner_fusion']
    dump(dict(method=formula,spec=spec,columns=cols,source_sha256=sha(__file__)),out/'MODELING_METHOD.json')
    dump(dict(control='V25',candidate_metrics=m,control_metrics=b,delta_KS=result['delta_KS'],delta_AUC=result['delta_AUC'],confidence_interval=interval,head_gap_candidate=float(np.mean([q['train_validation_head_KS_gap'] for q in checks])),head_gap_control=float(np.mean([q['train_validation_head_KS_gap'] for q in parent['folds']]))),out/'MATCHED_CONTROL.json')
    (out/'REPORT.md').write_text(f"# {ident}: {spec['label']}\n\n{formula}\n\nKS={m['KS']:.10f}; AUC={m['AUC']:.10f}; deltaKS={result['delta_KS']:+.10f}; paired95% CI={interval['delta_KS']}.\n\nSame5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.\n")
    if postprocess is not None:postprocess(out,result,tr,y,folds,reference,oof,frames)
    dump(result,out/'RESULT.json');dump(dict(status='COMPLETE',experiment=ident,metrics=m,target_reached=m['KS']>=.531),out/'STATUS.json')
    from master_log import generate
    generate();print('COMPLETE',ident,json.dumps(m),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    try:main(args.run)
    except Exception:
        dump(dict(status='FAILED',pid=os.getpid(),traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json');raise
