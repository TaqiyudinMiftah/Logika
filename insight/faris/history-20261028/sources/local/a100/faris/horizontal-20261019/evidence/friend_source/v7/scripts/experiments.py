"""Matched V7 offset ablations. Frozen Kumo and original spline scores unchanged."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import sys,json,time,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path.insert(0,str(ROOT/'scripts'))
from baseline import prepare,design,corrected_probability,objective,scores,dump,sha,ids_hash
sys.path.insert(0,str(Path(__file__).resolve().parent))
from features import div
from feature_policy import policy

PLAN=[
 dict(id='V00',families=[],label='Exact locked offset control',penalty=.003,strength=.5),
 dict(id='V01',families=['stress'],label='Financial stress cascade',penalty=.003,strength=.5),
 dict(id='V02',families=['dna'],label='Behavioral DNA',penalty=.003,strength=.5),
 dict(id='V03',families=['resilience'],label='Financial resilience',penalty=.003,strength=.5),
 dict(id='V04',families=['debt_safe'],label='Debt seeking and allocation trajectory',penalty=.003,strength=.5),
 dict(id='V05',families=['contradiction'],label='Financial contradiction',penalty=.003,strength=.5),
 dict(id='V06',families=['purpose'],label='Loan purpose consistency',penalty=.003,strength=.5),
 dict(id='V07',families=['composites'],label='Robust fragility and regime composites',penalty=.01,strength=.5),
 dict(id='V08',families=['stress','resilience'],label='Stress plus resilience',penalty=.01,strength=.5),
 dict(id='V09',families=['stress','dna','resilience','debt_safe','contradiction','purpose','composites'],label='All trusted incremental V7 L2 .01',penalty=.01,strength=.5),
 dict(id='V10',families=['stress','dna','resilience','debt_safe','contradiction','purpose','composites'],label='All trusted incremental V7 conservative L2 .03 shrink .25',penalty=.03,strength=.25),
 dict(id='V11',families=['stress'],only=['v7_stress_rejection_spending_shift','v7_stress_rejection_pretrend_adjusted'],label='Rejection event response isolation',penalty=.003,strength=.5),
 dict(id='V12',families=['dpd_research'],label='QUARANTINED DPD shape versus trusted allocation proxies',penalty=.003,strength=.5),
 dict(id='V13',families=['stress','dna','resilience','debt_safe','contradiction','purpose','composites','dpd_research'],label='QUARANTINED V7 plus DPD shape',penalty=.03,strength=.25)
]

for batch in ['extension','discovery','repayment','statistical','behavior','lifecycle']:
 batch_plan=V/(batch+'_plan.json')
 if batch_plan.exists():
  for extra in json.loads(batch_plan.read_text()):
   PLAN.append(dict(extra,fold_feature_root=batch+'/data',feature_catalog=batch+'/reports/FEATURE_CATALOG.csv',**{batch:True}))

def load_frames():
 raw=pd.read_parquet(V/'data/features_raw.parquet');assert raw.application_id.is_unique
 quarantine=V/'data/features_dpd_quarantined.parquet'
 if quarantine.exists():raw=raw.merge(pd.read_parquet(quarantine),on='application_id',validate='one_to_one')
 tr=pd.read_parquet(ROOT/'data/train_features.parquet');te=pd.read_parquet(ROOT/'data/test_features.parquet')
 assert set(tr.application_id).isdisjoint(te.application_id)
 return [x.merge(raw,on='application_id',how='left',validate='one_to_one') for x in [tr,te]]

def nuisance_array(d,kind):
 cols={'spend':['aux_declared_income','aux_account_income','aux_account_months','dependents','age'],
  'bureau':['aux_allocation_late_share','aux_allocation_count','aux_bureau_utilization','aux_bureau_freshness','aux_account_months'],
  'amount':['aux_declared_income','aux_account_income','dependents']}[kind]
 x=d[cols].to_numpy(float);x=np.sign(x)*np.log1p(abs(x))
 cat='employment_type' if kind=='spend' else 'loan_purpose'
 levels=['salaried','self_employed','variable_income'] if kind=='spend' else ['education','home_improvement','household','working_capital','medical']
 if kind!='bureau':x=np.column_stack([x,np.array([(d[cat]==level).to_numpy(float) for level in levels]).T])
 return x

def train_nuisance(x,y,kind):
 ok=np.isfinite(y)
 if ok.sum()<100:return dict(constant=float(np.nanmedian(y)) if ok.any() else 0.)
 model=HistGradientBoostingRegressor(loss='quantile' if kind=='spend' else 'squared_error',quantile=.5 if kind=='spend' else None,
  max_iter=80,max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=20,learning_rate=.05,early_stopping=False,random_state=31027)
 with threadpool_limits(limits=2):model.fit(x[ok],y[ok])
 return dict(model=model)

def nuisance_predict(model,x):
 if 'constant' in model:return np.full(len(x),model['constant'])
 with threadpool_limits(limits=2):return model['model'].predict(x)

def fit_fold_features(fold,tr,te):
 folder=V/'data'/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
 index=np.load(ROOT/'artifacts'/f'fold{fold}'/'indices.npz');hi=index['correction'];vi=index['validation'];ci=index['context']
 assert len(hi)==6000 and len(vi)==14000 and not np.intersect1d(hi,vi).size and not np.intersect1d(ci,hi).size
 h=tr.iloc[hi].copy();val=tr.iloc[vi].copy();test=te.copy();models={};isolation=[]
 for kind,target,newname in [('spend','aux_spend30','v7_stress_spending_income_anomaly'),('bureau','aux_bureau_score','v7_contradiction_score_allocation'),('amount','aux_loan_amount','v7_purpose_amount_surprise')]:
  y=h[target].to_numpy(float);y=np.log1p(y) if kind!='bureau' else y
  x=nuisance_array(h,kind);pv=np.full(len(h),np.nan);inner=[]
  for ti,qi in KFold(3,shuffle=True,random_state=31027+fold).split(x):
   assert not np.intersect1d(ti,qi).size
   model=train_nuisance(x[ti],y[ti],kind);pv[qi]=nuisance_predict(model,x[qi]);inner.append(dict(model=model,training_indices=ti,query_indices=qi))
   isolation.append(dict(kind=kind,fit_rows=len(ti),query_rows=len(qi),disjoint=True))
  model=train_nuisance(x,y,kind)
  for d,p in [(h,pv),(val,nuisance_predict(model,nuisance_array(val,kind))),(test,nuisance_predict(model,nuisance_array(test,kind)))]:
   actual=d[target].to_numpy(float);actual=np.log1p(actual) if kind!='bureau' else actual
   residual=actual-p
   if kind=='spend':residual=residual*div(d.aux_installment,d.aux_account_income)
   d[newname]=residual
  models[kind]=dict(final=model,inner=inner,fit_application_ids=h.application_id.tolist(),kind=kind,target=target)
 # References are unsupervised and fitted only on correction training; no test distribution used.
 refs={p:np.sort(h['aux_'+p].dropna().to_numpy(float)) for p in ['Bureau_A','Bureau_B','Bureau_C']}
 rawrisk=lambda d:np.column_stack([div(-d.aux_account_net,d.aux_account_income),d.aux_account_low_share,div(d.aux_installment,d.aux_account_income),d.aux_allocation_late_share])
 risk=rawrisk(h);median=np.nanmedian(risk,axis=0);iqr=np.nanquantile(risk,.75,axis=0)-np.nanquantile(risk,.25,axis=0);iqr=np.where(iqr>1e-8,iqr,1.)
 for d in [h,val,test]:
  ranks=np.column_stack([np.where(d['aux_'+p].notna(),np.searchsorted(refs[p],d['aux_'+p].to_numpy(float),side='right')/max(len(refs[p]),1),np.nan) for p in refs])
  counts=np.isfinite(ranks).sum(1);means=np.nansum(ranks,axis=1)/np.maximum(counts,1)
  d['v7_contradiction_bureau_provider_rank']=np.where(counts>=2,np.sqrt(np.nansum((ranks-means[:,None])**2,axis=1)/np.maximum(counts-1,1)),np.nan)
  z=np.clip((rawrisk(d)-median)/iqr,-5,5);count=np.isfinite(z).sum(1)
  d['v7_financial_fragility_index']=np.where(count>=2,np.nansum(z,axis=1)/np.maximum(count,1),np.nan)
  d['aux_ffi_observed_components']=count
 # Persist the actual feature inputs to each model, with identity/order preserved.
 keep=['application_id']+[c for c in h if c.startswith(('v7_','v7q_','aux_'))]
 for name,d in [('correction',h),('validation',val),('test',test)]:d[keep].to_parquet(folder/(name+'.parquet'),index=False)
 joblib.dump(dict(nuisance=models,provider_references=refs,fragility_median=median,fragility_iqr=iqr),folder/'nuisance.joblib',compress=3)
 dump(dict(status='PASS',fold=fold,correction_ids_sha256=ids_hash(tr,hi),validation_ids_sha256=ids_hash(tr,vi),
  no_default_target_in_nuisance=True,no_outer_validation_or_test_fit=True,inner_checks=isolation,
  constant_tenor_note='All selected terms are 12 months; amount nuisance model omits constant tenor',
  nuisance_training_rows=6000),folder/'AUDIT.json')

def combined(base,extra):
 assert base.application_id.tolist()==extra.application_id.tolist()
 cols=[c for c in extra if c not in base]
 return pd.concat([base.reset_index(drop=True),extra[cols].reset_index(drop=True)],axis=1)

def fit_head(frame,p,columns,penalty,strength):
 cats=frame[columns].select_dtypes(include=['object','string','category']).columns.tolist();nums=[c for c in columns if c not in cats]
 encoder=ColumnTransformer([
  ('smooth',Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('basis',SplineTransformer(n_knots=3,degree=2,knots='quantile',extrapolation='constant')),('scale',StandardScaler())]),nums),
  ('missing',MissingIndicator(features='all',sparse=False,error_on_new=False),nums),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
 with threadpool_limits(limits=2):basis=encoder.fit_transform(prepare(frame,columns,cats))
 offset=logit(np.clip(np.asarray(p,float),1e-6,1-1e-6));mean=float(offset.mean());sd=max(float(offset.std()),1e-8)
 x=np.column_stack([np.ones(len(frame)),(offset-mean)/sd,basis]);assert np.isfinite(x).all()
 y=frame.target.to_numpy(int);weight=np.ones(len(y))
 with threadpool_limits(limits=2):result=minimize(objective,np.zeros(x.shape[1]),args=(x,offset,y,weight,penalty,0.),method='L-BFGS-B',jac=True,options=dict(maxiter=500,ftol=1e-11,gtol=1e-7))
 loss,grad=objective(result.x,x,offset,y,weight,penalty,0.);assert result.success and np.max(abs(grad))<1e-4
 return dict(encoder=encoder,columns=columns,categorical=cats,offset_mean=mean,offset_sd=sd,coefficients=result.x,
  spec=dict(correction_strength=strength,penalty=penalty),positive_weight=1.,objective=loss,gradient_max_abs=float(np.max(abs(grad))),iterations=int(result.nit))

def metric(y,p):return dict(scores(y,p),logloss=float(log_loss(y,p)),brier=float(brier_score_loss(y,p)))

def bootstrap(y,baseline,pred,folds,reps=300):
 # Weight bootstrap on fixed sorted scores; exact ties evaluated together.
 def structure(p):
  order=np.argsort(p,kind='stable');end=np.r_[p[order][1:]!=p[order][:-1],True]
  return order,end
 structures=[structure(x) for x in [baseline,pred]]
 strata=[np.flatnonzero((folds==f)&(y==c)) for f in range(5) for c in [0,1]]
 rng=np.random.default_rng(192026);out=[]
 for repeat in range(reps):
  w=np.zeros(len(y),dtype=np.int32)
  for s in strata:w[s]=rng.multinomial(len(s),np.full(len(s),1/len(s)))
  totals=[np.sum(w[y==0]),np.sum(w[y==1])];values=[];aucs=[]
  for order,end in structures:
   pos=w[order]*(y[order]==1);neg=w[order]*(y[order]==0)
   cp=np.cumsum(pos)[end];cn=np.cumsum(neg)[end]
   values.append(float(np.max(abs(cp/totals[1]-cn/totals[0]))))
   dn=np.diff(np.r_[0,cn]);dp=np.diff(np.r_[0,cp]);aucs.append(float(np.sum(dp*(cn-.5*dn))/(totals[0]*totals[1])))
  out.append([values[0],values[1],values[1]-values[0],aucs[1],aucs[1]-aucs[0]])
 bounds=np.quantile(out,[.025,.975],axis=0)
 return dict(repetitions=reps,seed=192026,baseline_KS=bounds[:,0].tolist(),candidate_KS=bounds[:,1].tolist(),delta_KS=bounds[:,2].tolist(),candidate_AUC=bounds[:,3].tolist(),delta_AUC=bounds[:,4].tolist(),
  caveat='Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty')

def run(spec):
 started=time.time();name=spec['id'];out=V/'runs'/name;out.mkdir(parents=True,exist_ok=True)
 dump(dict(status='RUNNING',spec=spec,pid=os.getpid(),started=time.time()),out/'STATUS.json')
 lock=json.loads((ROOT/'BASELINE_LOCK.json').read_text());tr=pd.read_parquet(ROOT/'data/train_features.parquet');te=pd.read_parquet(ROOT/'data/test_features.parquet')
 sample=pd.read_csv(ROOT/'data/sample_submission.csv');catalog=pd.read_csv(V/spec.get('feature_catalog','reports/FEATURE_CATALOG.csv'))
 family_cols=catalog[catalog.family.isin(spec['families'])&catalog.status.isin(['IMPLEMENTED','FOLD_LOCAL','QUARANTINED','QUARANTINED_FOLD_LOCAL'])].feature_name.tolist()
 if 'only' in spec:family_cols=spec['only']
 support=[]
 if spec['families']:
  support=['aux_account_months','aux_orders90','aux_ecommerce_observed_days','aux_allocation_count']
  if 'stress' in spec['families']:support+=['aux_rejection_response_orders']
  if 'composites' in spec['families']:support+=['aux_regime_tested_domains','aux_ffi_observed_components']
 columns=list(dict.fromkeys(lock['correction']['features']+family_cols+support))
 oof=np.full(len(tr),np.nan);base=np.full(len(tr),np.nan);fold_ids=np.full(len(tr),-1);pt=np.zeros(len(te));checks=[]
 for fold in range(5):
  folder=ROOT/'artifacts'/f'fold{fold}';idx=np.load(folder/'indices.npz');hi,vi=idx['correction'],idx['validation']
  feature_root=V/spec.get('fold_feature_root','data')
  h=combined(tr.iloc[hi],pd.read_parquet(feature_root/f'fold{fold}'/'correction.parquet'))
  val=combined(tr.iloc[vi],pd.read_parquet(feature_root/f'fold{fold}'/'validation.parquet'))
  test=combined(te,pd.read_parquet(feature_root/f'fold{fold}'/'test.parquet'))
  kh,kv,kt=[pd.read_parquet(folder/f'kumo_{k}.parquet') for k in ['holdout','oof','test']]
  for d,c in [(h,kh),(val,kv),(test,kt)]:assert d.application_id.tolist()==c.application_id.tolist()
  old=joblib.load(folder/'correction.joblib');head=old if name=='V00' else fit_head(h,kh.prediction,columns,spec['penalty'],spec['strength'])
  ph,dh=corrected_probability(head,h,kh.prediction);pv,dv=corrected_probability(head,val,kv.prediction);ptest,dt=corrected_probability(head,test,kt.prediction)
  spv=pd.read_parquet(folder/'spline_oof.parquet');spt=pd.read_parquet(folder/'spline_test.parquet')
  assert spv.application_id.tolist()==val.application_id.tolist() and spt.application_id.tolist()==test.application_id.tolist()
  p=.75*pv+.25*spv.prediction.to_numpy();tp=.75*ptest+.25*spt.prediction.to_numpy()
  locked=pd.read_parquet(folder/'final_oof.parquet');assert locked.application_id.tolist()==val.application_id.tolist()
  if name=='V00':assert np.max(abs(p-locked.prediction))<2e-14
  base[vi]=locked.prediction;oof[vi]=p;fold_ids[vi]=fold
  pt+=pd.Series(tp,index=te.application_id).reindex(sample.application_id).to_numpy()/5
  stats=metric(val.target,p);bstat=metric(val.target,locked.prediction)
  check=dict(fold=fold,metrics=stats,baseline=bstat,delta_KS=stats['KS']-bstat['KS'],delta_AUC=stats['AUC']-bstat['AUC'],
   correction_rows=len(h),parameters=len(head['coefficients']),head_training_KS=scores(h.target,ph)['KS'],heldout_corrected_Kumo_KS=scores(val.target,pv)['KS'],
   train_validation_head_KS_gap=scores(h.target,ph)['KS']-scores(val.target,pv)['KS'],mean_abs_correction=float(np.mean(abs(dv))),
   P99_abs_correction=float(np.quantile(abs(dv),.99)),optimizer_gradient_max=head['gradient_max_abs'],optimizer_iterations=head['iterations'],
   train_ids_sha256=ids_hash(tr,hi),validation_ids_sha256=ids_hash(tr,vi))
  checks.append(check);joblib.dump(head,out/f'fold{fold}_correction.joblib',compress=3)
  pd.DataFrame(dict(application_id=val.application_id,prediction=p,correction_logit=dv)).to_parquet(out/f'fold{fold}_validation.parquet',index=False)
  dump(dict(status='RUNNING',folds_complete=fold+1,spec=spec,pid=os.getpid()),out/'STATUS.json')
  print(name,'fold',fold,'KS',stats['KS'],'delta',check['delta_KS'],flush=True)
 assert np.isfinite(oof).all() and np.isfinite(pt).all() and np.all((pt>0)&(pt<1))
 submission=V/'submissions'/f'submission_{name}_goal054_offset_v7.csv';submission.parent.mkdir(exist_ok=True)
 pd.DataFrame(dict(application_id=sample.application_id,default_probability=pt)).to_csv(submission,index=False)
 digest=sha(submission)
 if name=='V00':
  original=ROOT/'submissions'/lock['submission_file'];saved=pd.read_csv(original)
  assert saved.application_id.tolist()==sample.application_id.tolist() and np.max(abs(pt-saved.default_probability))<2e-14
  submission.write_bytes(original.read_bytes());digest=sha(submission);assert digest==lock['submission_sha256']
 pd.DataFrame(dict(application_id=tr.application_id,target=tr.target,fold=fold_ids,prediction=oof,baseline_prediction=base)).to_csv(out/'OOF.csv',index=False)
 performance=metric(tr.target,oof);bperf=metric(tr.target,base);interval=bootstrap(tr.target.to_numpy(int),base,oof,fold_ids)
 gains=[x['delta_KS'] for x in checks];foldKS=[x['metrics']['KS'] for x in checks];wins=sum(g>0 for g in gains)
 evidence='PROMISING_REQUIRES_INDEPENDENT_VALIDATION' if interval['delta_KS'][0]>0 and wins>=4 and performance['AUC']>=bperf['AUC']-.0005 else 'INCONCLUSIVE_OR_NEGATIVE'
 if name=='V00':evidence='EXACT_CONTROL'
 snapshot_dependent='dpd_research' in spec['families'] or spec.get('quarantined',False) or spec.get('snapshot_dependent',False)
 fields=spec.get('snapshot_fields') or (['max_days_past_due'] if snapshot_dependent else [])
 dpd_used=any('past_due' in c for c in fields)
 quarantined=(spec.get('quarantined',False) or 'dpd_research' in spec['families']) and not (dpd_used and 'amount_paid' not in fields and policy().get('allow_DPD_without_quarantine',False))
 if quarantined:evidence='QUARANTINED_SNAPSHOT_PROVENANCE_UNVERIFIED'
 result=dict(status='COMPLETE',experiment=name,spec=spec,columns=columns,new_features=family_cols,support_features=support,OOF=performance,baseline=bperf,
  delta_KS=performance['KS']-bperf['KS'],delta_AUC=performance['AUC']-bperf['AUC'],mean_fold_KS=float(np.mean(foldKS)),std_fold_KS=float(np.std(foldKS,ddof=1)),
  delta_mean_fold_KS=float(np.mean(gains)),fold_wins=wins,folds=checks,confidence_interval=interval,evidence=evidence,
  overfitting_assessment=dict(development_folds_reused=True,independent_holdout=False,hyperparameters_prespecified=True,retrained_seeds=1,
   extra_head_features=len(columns)-12,selection_corrected_CI=False,public_score_unknown=name!='V00',
   risk='HIGH_SEARCH_MULTIPLICITY; reduce only with independent time/seed evidence',
   diagnostic='Head training KS includes fitted correction, so it is optimistic; compare heldout evidence and baseline, not in-sample score'),
  frozen_Kumo_fraction=1.,backbone_retrained=False,original_spline_unchanged=True,all_outer_indices_identical=True,
  point_in_time_certified=False,inherited_baseline_snapshot_uncertainty=True,
  new_features_DPD_used=dpd_used,effective_quarantined=quarantined,feature_policy=policy(),
  new_features_snapshot_dependent=snapshot_dependent,new_snapshot_fields=spec.get('snapshot_fields',[]),
  submission=str(submission.relative_to(ROOT)),submission_sha256=digest,public_score=lock['public_score'] if digest==lock['submission_sha256'] else None,
  elapsed_seconds=time.time()-started,source_sha256=sha(__file__),feature_matrix_sha256=sha(V/'data/features_raw.parquet'))
 if spec.get('extension'):
  result['extension_feature_matrix_sha256']=sha(V/'extension/data/raw_features.parquet')
  result['extension_source_sha256']=sha(V/'scripts/extension_features.py')
 if spec.get('discovery'):
  result['discovery_feature_matrix_sha256']=sha(V/'discovery/data/raw_features.parquet')
  result['discovery_source_sha256']=sha(V/'scripts/discovery_features.py')
 if spec.get('repayment'):
  result['repayment_feature_matrix_sha256']=sha(V/'repayment/data/raw_features.parquet')
  result['repayment_source_sha256']=sha(V/'scripts/repayment_features.py')
 if spec.get('statistical'):
  result['statistical_feature_matrix_sha256']=sha(V/'statistical/data/raw_features.parquet')
  result['statistical_source_sha256']=sha(V/'scripts/statistical_features.py')
  result['statistical_transition_events_sha256']=sha(V/'statistical/data/transitions.parquet')
 if spec.get('behavior'):
  result['behavior_feature_matrix_sha256']=sha(V/'behavior/data/raw_features.parquet')
  result['behavior_source_sha256']=sha(V/'scripts/behavior_features.py')
  result['behavior_size_bins_sha256']=sha(V/'behavior/data/size_bins.parquet')
 if spec.get('lifecycle'):
  result['lifecycle_source_sha256']=sha(V/'scripts/lifecycle_features.py')
  result['lifecycle_feature_matrix_sha256']=sha(V/'lifecycle/data/raw_features.parquet')
 dump(result,out/'RESULT.json');dump(dict(status='COMPLETE',submission_sha256=digest),out/'STATUS.json');print(json.dumps({k:result[k] for k in ['experiment','OOF','delta_KS','fold_wins','evidence']}),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare-fold',type=int);p.add_argument('--run');args=p.parse_args()
 if args.prepare_fold is not None:fit_fold_features(args.prepare_fold,*load_frames())
 elif args.run:run(next(s for s in PLAN if s['id']==args.run))
 else:print(json.dumps(PLAN,indent=2))
