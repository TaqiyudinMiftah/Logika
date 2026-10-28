import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,time,joblib
import numpy as np,pandas as pd
import baseline as b,feature_head,spline
from gpu_backend import Engine
from maturity_features import MaturityFeatures
R=Path(__file__).resolve().parents[1];A=R.parent;INPUT=A/'ranking-pipeline-20261020/input';RAW=A/'exploration-12h-20261017/raw';SELECTED=A/'shopping-periodicity-fe-20261021'
def read(p):return json.loads(Path(p).read_text())
def main(f):
 start=time.time();plan=read(R/'plan.json');assert time.time()<plan['refit_deadline']and read(R/'PREPARED.json')['status']=='passed';root=R/f'models/fold{f}';root.mkdir(parents=True,exist_ok=False);ix=np.load(R/f'input/artifacts/fold{f}/indices.npz');assert len(ix['context'])==64000 and len(ix['correction'])==6000 and len(ix['train'])==70000
 for name,h in plan['source_sha256'].items():assert b.sha(R/'source'/name)==h
 lock=read(INPUT/'BASELINE_LOCK.json');checkpoint=A/'ranking-pipeline-20261020/assets/classifier.pt';assert b.sha(checkpoint)==lock['Kumo']['checkpoint_sha256'];feature_root=R/f'feature_build/fold{f}';assert b.sha(feature_root/'features.parquet')==read(feature_root/'complete.json')['feature_table_sha256'];frame=pd.read_parquet(feature_root/'features.parquet');assert len(frame)==100000 and'target'not in frame
 apps=pd.read_parquet(R/'applications.parquet');assert frame.application_id.tolist()==apps.application_id.tolist();y=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['target']).target.to_numpy(int);columns=read(A/f'kumo-coverage-placement-20261021/conditions/fold{f}/control90/boundary.json')['feature_columns'];assert len(columns)==90
 # Freeze the exact canonical context before inference. Attention caches are
 # derived indexes; all optimized/head/spline parameters are separately saved.
 context=frame.iloc[ix['context']][['application_id']+columns].copy();context['target']=y[ix['context']];joblib.dump(dict(context=context,columns=columns,seed=2026+f,num_estimators=8,checkpoint_sha256=b.sha(checkpoint),kind='frozen_Kumo_large_full_refit_context'),root/'kumo_model.joblib',compress=3);canonical=joblib.load(root/'kumo_model.joblib');pd.testing.assert_frame_equal(context,canonical['context']);assert set(context.application_id).isdisjoint(frame.application_id.iloc[ix['correction']])and set(context.application_id).isdisjoint(frame.application_id.iloc[70000:])
 query_rows=np.r_[ix['correction'],np.arange(70000,100000)];query=frame.iloc[query_rows][['application_id']+columns].copy();print(json.dumps(dict(stage='Kumo_full_refit_start',fold=f,context=64000,correction=6000,test=30000,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))),flush=True)
 engine=Engine(checkpoint,columns,canonical['context']);raw=engine.fit_predict(np.arange(64000),query,2026+f,cache_path=None);assert len(raw)==36000 and np.isfinite(raw).all()and engine.last_prediction_replay_error==0.;np.savez_compressed(root/'raw_predictions.npz',rows=query_rows,application_id=query.application_id.to_numpy(str),prediction=raw);del engine
 selected_head=joblib.load(SELECTED/f'models/true/fold{f}/head.joblib');selected_sp=joblib.load(SELECTED/f'models/true/fold{f}/spline.joblib');assert len(selected_head['columns'])==29 and len(selected_sp['columns'])==88
 cov=pd.read_parquet(R/'coverage.parquet').set_index('application_id').loc[frame.application_id];spectral=pd.read_parquet(SELECTED/'true_features.parquet').set_index('application_id').loc[frame.application_id];assert b.sha(SELECTED/'true_features.parquet')==read(R/'PREPARED.json')['selected_feature_sha256']
 for c in selected_head['columns']:
  if c in cov:frame[c]=cov[c].to_numpy()
  elif c in spectral:frame[c]=spectral[c].to_numpy()
 history=pd.read_csv(RAW/'repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due']);factory=MaturityFeatures().fit(apps,history,apps.application_id.iloc[ix['context']]);factory.save(root/'maturity_transformer.json');mature=factory.transform(apps,history).set_index('application_id').loc[frame.application_id];again=MaturityFeatures.load(root/'maturity_transformer.json').transform(apps,history).set_index('application_id').loc[frame.application_id];pd.testing.assert_frame_equal(mature,again,check_exact=True)
 for c in ['re_late_1_posterior','re_late_30_posterior']:frame[c]=mature[c].to_numpy()
 for c in selected_sp['categorical']:frame[c]=frame[c].astype(object)
 train_head=frame.iloc[ix['correction']].copy();train_head['target']=y[ix['correction']];head=feature_head.fit(train_head,raw[:6000],selected_head['columns'],.003,.5);alltrain=frame.iloc[:70000].copy();alltrain['target']=y;sp=spline.fit(alltrain,selected_sp['columns']);assert head['spec']==selected_head['spec']and sp['resolved_parameters']==selected_sp['resolved_parameters'];joblib.dump(head,root/'head.joblib',compress=3);joblib.dump(sp,root/'spline.joblib',compress=3)
 test=frame.iloc[70000:].copy();assert'target'not in test;test[['application_id']+list(dict.fromkeys(head['columns']+sp['columns']))].to_parquet(root/'test_features.parquet',index=False)
 def predict(h,s):return .75*b.corrected_probability(h,test,raw[6000:])[0]+.25*b.predict_spline(s,test)
 pred=predict(head,sp);assert np.array_equal(pred,predict(joblib.load(root/'head.joblib'),joblib.load(root/'spline.joblib')));assert np.isfinite(pred).all()and((pred>=0)&(pred<=1)).all();np.savez_compressed(root/'test_predictions.npz',application_id=test.application_id.to_numpy(str),prediction=pred)
 b.dump(dict(status='passed',fold=f,context_rows=64000,correction_rows=6000,spline_rows=70000,every_training_row_used=True,seed=2026+f,num_estimators=8,head_inputs=29,spline_inputs=88,head_iterations=head['iterations'],head_gradient_max_abs=head['gradient_max_abs'],same_model_hyperparameters=True,test_labels_available=False,serialized_models_replay_exact=True,Kumo_first_batch_replay_exact=True,checkpoint_sha256=b.sha(checkpoint),seconds=time.time()-start,files={p.name:b.sha(p)for p in root.iterdir()if p.is_file()}),root/'complete.json');print(json.dumps(dict(stage='full_refit_complete',fold=f,seconds=time.time()-start)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);main(p.parse_args().fold)
