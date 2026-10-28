"""Incremental CPU-only evidence audit. Never reads outer validation labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,sys
import numpy as np,pandas as pd,joblib
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
import baseline as b,feature_head
def read(p):return json.loads(Path(p).read_text())
def main():
 plan=read(R/'training-plan.json');columns=read(R/'columns.json');path=R/'CONTENT_AUDIT.json'
 previous=read(path)if path.exists()else dict(checks={})
 for fold in range(5):
  root=R/f'runs/fold{fold}';complete=root/'complete.json'
  if not complete.exists():continue
  receipt=b.sha(complete)
  if str(fold)in previous['checks']and previous['checks'][str(fold)]['completion_sha256']==receipt:continue
  done=read(complete)
  for n,h in done['files'].items():assert b.sha(root/n)==h
  data=R/f'data/fold{fold}';manifest=read(data/'manifest.json')
  for n,h in manifest['files'].items():assert b.sha(data/n)==h
  for n,h in plan['worker_source_sha256'].items():assert b.sha(R/'source'/n)==h
  model=joblib.load(root/'context_model.joblib');context=pd.read_parquet(data/'context.parquet');q=pd.read_parquet(data/'query.parquet');ix=np.load(data/'indices.npz')
  assert model['kind']=='frozen_context_ICL_model_v1'and model['seed']==2026+fold and model['num_estimators']==8 and model['dtype']=='float16'
  assert model['columns']==columns and len(columns)==90 and model['checkpoint_sha256']==plan['checkpoint_sha256']
  assert model['engine_sha256']==b.sha(R/'source/gpu_backend.py')and model['model_code_sha256']==b.sha(R/'source/context_model.py')
  pd.testing.assert_frame_equal(model['context'],context[['application_id','target']+columns],check_exact=True)
  assert len(context)==50000 and len(q)==20000 and'target'not in q and set(context.application_id).isdisjoint(q.application_id)
  assert len(np.unique(np.r_[ix['context'],ix['correction'],ix['validation']]))==70000
  raw=np.load(root/'raw.npz');assert np.array_equal(raw['rows'],np.r_[ix['correction'],ix['validation']])and raw['application_id'].tolist()==q.application_id.tolist()
  assert np.isfinite(raw['prediction']).all()and np.all((raw['prediction']>0)&(raw['prediction']<1));error=float(abs(raw['prediction']-raw['replayed_prediction']).max());assert error<2e-5
  labels=np.load(data/'correction_labels.npy').astype(np.int64);assert len(labels)==6000
  head=joblib.load(root/'head.joblib');assert len(head['columns'])==18 and not{'target','application_id','customer_id'}&set(head['columns'])
  x,offset=b.design(head,q.iloc[:6000],raw['prediction'][:6000]);loss,gradient=b.objective(head['coefficients'],x,offset,labels,np.ones(6000),.003)
  assert abs(loss-head['objective'])<1e-12 and max(abs(gradient))<1e-4
  independently_refitted=feature_head.fit(q.iloc[:6000].assign(target=labels),raw['prediction'][:6000],head['columns'],.003,.5)
  pr=b.corrected_probability(head,q.iloc[6000:],raw['prediction'][6000:])[0];refit=b.corrected_probability(independently_refitted,q.iloc[6000:],raw['prediction'][6000:])[0]
  head_error=float(abs(pr-refit).max());assert head_error<1e-12
  poisoned=q.iloc[6000:].copy();poisoned['target']=999;poisoned.application_id='POISONED';pp=b.corrected_probability(head,poisoned,raw['prediction'][6000:])[0];assert np.array_equal(pr,pp)
  sp=np.load(data/'spline_validation.npz');assert np.array_equal(sp['rows'],ix['validation']);final=.75*pr+.25*sp['prediction'];val=np.load(root/'validation.npz');assert np.array_equal(val['rows'],ix['validation'])and val['application_id'].tolist()==q.application_id.iloc[6000:].tolist()
  assert np.array_equal(final,val['prediction'])and abs(val['prediction']-val['replayed_prediction']).max()<2e-5
  previous['checks'][str(fold)]=dict(completion_sha256=receipt,context_bytes_and_labels_verified=True,source_checkpoint_and_seed_verified=True,complete_raw_model_cold_replay_max_error=error,head_objective_gradient_rebuilt=True,head_refit_prediction_max_error=head_error,outer_label_and_ID_poisoning_invariance=True,validation_rows=14000,outer_labels_opened=False)
 previous.update(status='passed_for_available_completed_folds',completed_folds=len(previous['checks']),outer_labels_opened=False,audit_source_sha256=b.sha(Path(__file__)),observed_at=time.time());b.dump(previous,path)
 print(json.dumps(dict(status=previous['status'],completed_folds=previous['completed_folds'],outer_labels_opened=False)),flush=True)
if __name__=='__main__':main()
