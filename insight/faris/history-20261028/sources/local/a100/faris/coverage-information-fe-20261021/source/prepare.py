import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np,pandas as pd,joblib
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;INPUT=A/'ranking-pipeline-20261020/input';DATA=A/'kumo-temperature-adaptation-20261021/full_data';FIX=A/'fold-local-priors-20261021';PRIOR=A/'repayment-severity-fe-20261021';SPLINE=A/'maturity-placement-fe-20261021';COVER=A/'coverage-transfer-20261021'
DROP=['v4_price_all_mean','v4_price_180d_mean','v4_price_90d_mean','v5_price_pooled_all_mean']
KEEP=['v4_price_all_equal_order_mean','v4_price_180d_equal_order_mean','v4_price_90d_equal_order_mean','v5_price_pooled_all_equal_order_mean']
ADD=['risk_conditional_negative_months','risk_conditional_nonnegative_months','aux_orders90','aux_allocation_count']
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not(R/'DATA_READY.json').exists();plan=read(R/'plan.json');assert time.time()<plan['original_deadline']-10800
 for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id']);lock=read(INPUT/'BASELINE_LOCK.json');cols=[ADD[DROP.index(c)]if c in DROP else c for c in lock['Kumo']['features']]
 assert len(cols)==len(set(cols))==90 and set(ADD).isdisjoint(lock['Kumo']['features'])and set(KEEP)<=set(cols)
 assert read(COVER/'INDEPENDENT_AUDIT.json')['status']=='passed';coverage=pd.read_parquet(COVER/'data/features.parquet').set_index('application_id');assert coverage.index.is_unique
 audits=[]
 for f in range(5):
  assert time.time()<plan['original_deadline']-10800;source=DATA/f'fold{f}';manifest=read(source/'manifest.json')
  for n,h in manifest['files'].items():assert b.sha(source/n)==h
  ix=np.load(source/'indices.npz');original_ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz')
  for role in ['context','correction','validation']:assert np.array_equal(ix[role],original_ix[role])
  c=pd.read_parquet(source/'context.parquet');q=pd.read_parquet(source/'query.parquet');y=np.load(source/'correction_labels.npy').astype(np.int64);assert len(c)==50000 and len(q)==20000 and len(y)==6000 and'target'not in q
  assert c.application_id.tolist()==original.application_id.iloc[ix['context']].tolist()and q.application_id.tolist()==original.application_id.iloc[np.r_[ix['correction'],ix['validation']]].tolist()and set(c.application_id).isdisjoint(q.application_id)
  checks=[]
  for drop,keep,add in zip(DROP,KEEP,ADD):
   xx=c[drop].to_numpy(float);yy=c[keep].to_numpy(float);finite=np.isfinite(xx)&np.isfinite(yy);corr=float(np.corrcoef(xx[finite],yy[finite])[0,1]);assert corr>=.98 and np.array_equal(np.isfinite(xx),np.isfinite(yy))
   for frame in [c,q]:assert np.array_equal(frame[add].to_numpy(),coverage.loc[frame.application_id,add].to_numpy(),equal_nan=True)
   checks.append(dict(replaced=drop,retained=keep,added=add,context_Pearson=corr,missing_masks_equal=True,source_coverage_hash=b.sha(COVER/'data/features.parquet')))
  # The same correction-stage maturity means as the supported lead. No new
  # population fitting or field selection occurs in this feature packet.
  pp=read(PRIOR/f'fold{f}/transformer.json');assert b.sha(PRIOR/f'fold{f}/features.parquet')==pp['feature_hash'];means=pd.read_parquet(PRIOR/f'fold{f}/features.parquet').set_index('application_id').loc[q.application_id]
  for t in [1,30]:q[f're_late_{t}_posterior']=means[f'matched_beta_late_{t}'].to_numpy(float)
  dest=R/f'data/fold{f}';dest.mkdir(parents=True);c[['application_id','target']+cols].to_parquet(dest/'context.parquet',index=False);q.to_parquet(dest/'query.parquet',index=False);np.save(dest/'correction_labels.npy',y);np.savez_compressed(dest/'indices.npz',context=ix['context'],correction=ix['correction'],validation=ix['validation'])
  spline_root=SPLINE/f'fold{f}';done=read(spline_root/'complete.json')
  for n,h in done['files'].items():assert b.sha(spline_root/n)==h
  sm=joblib.load(spline_root/'spline.joblib');repaired=FIX/f'fold{f}';assert b.sha(repaired/'features.parquet')==read(repaired/'complete.json')['feature_table_sha256']
  val=pd.read_parquet(repaired/'features.parquet',columns=['application_id']+sm['columns']).set_index('application_id').loc[q.application_id.iloc[6000:]].reset_index()
  for t in [1,30]:val[f're_late_{t}_posterior']=means[f'matched_beta_late_{t}'].to_numpy()[6000:]
  sp=b.predict_spline(sm,val);headroot=PRIOR/f'heads/matched_maturity/fold{f}';head=joblib.load(headroot/'model.joblib');raw=np.load(A/f'kumo-coverage-placement-20261021/conditions/fold{f}/control90/raw_predictions.npz')['prediction'][6000:];final=.75*b.corrected_probability(head,q.iloc[6000:],raw)[0]+.25*sp
  saved=np.load(spline_root/'validation.npz');assert np.array_equal(saved['rows'],ix['validation'])and np.max(abs(final-saved['both']))<2e-14
  np.savez_compressed(dest/'spline_validation.npz',rows=ix['validation'],prediction=sp)
  b.dump(dict(fold=f,feature_substitutions=checks,core_input_width=90,prior_context_ids_sha256=b.ids_hash(original,ix['context']),query_labels_staged=False,outer_validation_labels_staged=False,correction_labels_separate=True,spline_reuse_source=str(spline_root/'spline.joblib'),spline_source_sha256=b.sha(spline_root/'spline.joblib'),maturity_only_pipeline_replay_max_error=float(abs(final-saved['both']).max())),dest/'boundary.json')
  b.dump(dict(status='complete',fold=f,files={p.name:b.sha(p)for p in dest.iterdir()if p.is_file()}),dest/'manifest.json');audits.append(dict(fold=f,substitutions=checks));print(json.dumps(dict(stage='data_ready',fold=f)),flush=True)
 b.dump(cols,R/'columns.json');b.dump(dict(status='passed',folds=audits,query_default_labels_staged=False,core_width_unchanged=True,source_coverage_independently_verified=True,shared_spline_replayed=True),R/'DATA_READY.json')
if __name__=='__main__':main()
