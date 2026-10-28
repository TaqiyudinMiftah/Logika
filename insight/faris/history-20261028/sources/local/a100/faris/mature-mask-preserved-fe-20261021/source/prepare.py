import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time,json,gc
import numpy as np,pandas as pd,joblib
import baseline as b,spline
R=Path(__file__).resolve().parents[1];A=R.parent;INPUT=A/'ranking-pipeline-20261020/input';RAW=A/'exploration-12h-20261017/raw';DATA=A/'kumo-temperature-adaptation-20261021/full_data';FIX=A/'fold-local-priors-20261021';PRIOR=A/'repayment-severity-fe-20261021';MAIN=A/'kumo-coverage-placement-20261021'
SPECS=[('v3_repay_all_late_'+str(t)+'_posterior_24',t,None)for t in [1,7,30,90]]+[('v3_repay_180d_late_1_posterior_24',1,180)]
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'plan.json')['original_deadline']-10800 and not(R/'STOP').exists()
def main():
 check();assert not(R/'DATA_READY.json').exists();plan=read(R/'plan.json')
 for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);lock=read(INPUT/'BASELINE_LOCK.json');cols=lock['Kumo']['features'];assert len(cols)==90 and all(c in cols and c in lock['original_spline']['features']for c,_,_ in SPECS)
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True);apps.decision_datetime=pd.to_datetime(apps.decision_datetime);assert apps.application_id.iloc[:70000].tolist()==original.application_id.tolist()
 rp=pd.read_csv(RAW/'repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'],parse_dates=['due_date']).merge(apps,on='customer_id',validate='many_to_one');rp['age']=(rp.decision_datetime-rp.due_date).dt.total_seconds()/86400;assert rp.age.gt(0).all()and not rp.duplicated(['customer_id','installment_id']).any()
 audits=[]
 for f in range(5):
  check();ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');fit_ids=set(original.customer_id.iloc[ix['context']]);changed=apps[['application_id','customer_id']].copy();evidence={}
  for c,t,w in SPECS:
   part=rp[rp.age.ge(t)&(True if w is None else rp.age.le(w))].copy();part['late']=part.max_days_past_due.ge(t).astype(int);training=part[part.customer_id.isin(fit_ids)];prevalence=float(training.late.mean());g=part.groupby('customer_id').late.agg(['sum','count']).reindex(apps.customer_id).fillna(0)
   assert np.all(g['sum']<=g['count']);values=(g['sum']+24*prevalence)/(g['count']+24);changed[c]=values.to_numpy(float)
   # Independently rebuild training prevalence from raw masks and positive counts.
   mask=rp.customer_id.isin(fit_ids)&rp.age.ge(t)
   if w is not None:mask &=rp.age.le(w)
   assert abs(prevalence-float(rp.loc[mask,'max_days_past_due'].ge(t).sum())/int(mask.sum()))<1e-14
   evidence[c]=dict(threshold=t,window=w,prior=prevalence,shrinkage=24,eligible_events=len(part),ineligible_for_threshold=int((rp.age.lt(t)&(True if w is None else rp.age.le(w))).sum()),zero_eligible_customers=int(g['count'].eq(0).sum()),fit_events=len(training),fit_ids_sha256=b.ids_hash(original,ix['context']),formula='(eligible late count +24*contexteligibleprevalence)/(eligible count+24)')
  # The only difference from the completed eligible-denominator trial:
  # preserve each legacy field's original missing-value mask.
  old_missing=pd.read_parquet(FIX/f'fold{f}/features.parquet',columns=['application_id']+[c for c,_,_ in SPECS]).set_index('application_id').loc[changed.application_id]
  parent=A/'mature-denominator-fe-20261021'/f'data/fold{f}'
  parent_values=pd.read_parquet(parent/'changed_features.parquet').set_index('application_id').loc[changed.application_id]
  for c,_,_ in SPECS:
   assert np.array_equal(changed[c].to_numpy(),parent_values[c].to_numpy())
   changed[c]=changed[c].mask(old_missing[c].isna().to_numpy())
   evidence[c]['original_missing_mask_preserved']=True
   evidence[c]['original_missing_count']=int(old_missing[c].isna().sum())
  dest=R/f'data/fold{f}';dest.mkdir(parents=True);changed.to_parquet(dest/'changed_features.parquet',index=False)
  repaired=FIX/f'fold{f}';assert b.sha(repaired/'features.parquet')==read(repaired/'complete.json')['feature_table_sha256']
  required=list(dict.fromkeys(cols+lock['original_spline']['features']+lock['correction']['features']))
  frame=pd.read_parquet(repaired/'features.parquet',columns=['application_id']+required).set_index('application_id').loc[original.application_id].reset_index();assert'target'not in frame
  packet=DATA/f'fold{f}';man=read(packet/'manifest.json')
  for n,h in man['files'].items():assert b.sha(packet/n)==h
  ctx=pd.read_parquet(packet/'context.parquet');query=pd.read_parquet(packet/'query.parquet');labels=np.load(packet/'correction_labels.npy').astype(np.int64);assert'target'not in query
  assert ctx.application_id.tolist()==original.application_id.iloc[ix['context']].tolist()and query.application_id.tolist()==original.application_id.iloc[np.r_[ix['correction'],ix['validation']]].tolist()
  y=pd.concat([ctx.set_index('application_id').target,pd.Series(labels,index=query.application_id.iloc[:6000])]);assert len(y)==56000 and y.index.is_unique and set(y.index).isdisjoint(query.application_id.iloc[6000:])
  oldsp=joblib.load(MAIN/f'shared/fold{f}/spline.joblib')
  for c in oldsp['categorical']:frame[c]=frame[c].astype(object)
  oldpred=b.predict_spline(oldsp,frame.iloc[ix['validation']]);sp=np.load(MAIN/f'shared/fold{f}/validation.npz');assert np.array_equal(sp['rows'],ix['validation'])and np.max(abs(oldpred-sp['prediction']))<2e-14
  changes=changed.set_index('application_id').loc[frame.application_id]
  for c,_,_ in SPECS:frame[c]=changes[c].to_numpy(float)
  previous=read(PRIOR/f'fold{f}/transformer.json');assert b.sha(PRIOR/f'fold{f}/features.parquet')==previous['feature_hash'];beta=pd.read_parquet(PRIOR/f'fold{f}/features.parquet').set_index('application_id').loc[frame.application_id]
  for t in [1,30]:frame[f're_late_{t}_posterior']=beta[f'matched_beta_late_{t}'].to_numpy(float)
  # Keep all six V25 coverage fields unchanged, matched by application rather than position.
  extras=[c for c in query.columns if c not in frame.columns]
  lookup=pd.concat([ctx.set_index('application_id'),query.set_index('application_id')])
  for c in extras:
   if c not in ['target','application_id']:frame[c]=lookup.loc[frame.application_id,c].to_numpy()
  context=frame.iloc[ix['context']][['application_id']+cols].copy();context['target']=y.loc[context.application_id].to_numpy(np.int64)
  q=frame.iloc[np.r_[ix['correction'],ix['validation']]].copy();assert'target'not in q and set(context.application_id).isdisjoint(q.application_id)
  context.to_parquet(dest/'context.parquet',index=False);q.to_parquet(dest/'query.parquet',index=False);np.save(dest/'correction_labels.npy',labels)
  np.savez_compressed(dest/'indices.npz',context=ix['context'],correction=ix['correction'],validation=ix['validation'])
  train=frame.iloc[ix['train']].copy();train['target']=y.loc[train.application_id].to_numpy(np.int64);newsp=spline.fit(train,oldsp['columns']);assert newsp['resolved_parameters']==oldsp['resolved_parameters']and newsp['model'].coef_.shape==oldsp['model'].coef_.shape
  joblib.dump(newsp,dest/'spline.joblib',compress=3);pred=b.predict_spline(newsp,frame.iloc[ix['validation']]);replay=b.predict_spline(joblib.load(dest/'spline.joblib'),frame.iloc[ix['validation']]);assert np.array_equal(pred,replay);np.savez_compressed(dest/'spline_validation.npz',rows=ix['validation'],prediction=pred)
  b.dump(dict(fold=f,evidence=evidence,prior_estimator_receives_default_labels=False,outer_validation_labels_staged=False,correction_labels_staged_separately=True,source_raw_hash=b.sha(RAW/'repayment_history.csv'),prior_context_ids_sha256=b.ids_hash(original,ix['context']),old_spline_replay=True,new_spline_replay_exact=True),dest/'boundary.json')
  b.dump(dict(status='complete',fold=f,files={p.name:b.sha(p)for p in dest.iterdir()if p.is_file()}),dest/'manifest.json');audits.append(dict(fold=f,features=evidence));print(json.dumps(dict(stage='data_ready',fold=f)),flush=True);gc.collect()
 b.dump(cols,R/'columns.json');b.dump(dict(status='passed',folds=5,feature_lineage=audits,outer_validation_labels_staged=False,correction_labels_staged_separately=True,all_shared_priors_context_only=True,architecture_frozen=True,original_missing_masks_preserved=True),R/'DATA_READY.json')
if __name__=='__main__':main()
