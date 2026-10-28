"""Independent saved-model reconstruction and full OOF metric verification."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'));import baseline as b
A=R.parent;INPUT=A/'ranking-pipeline-20261020/input';DATA=A/'kumo-temperature-adaptation-20261021/full_data'
def read(p):return json.loads(p.read_text())
def main():
 assert read(R/'TRAINING_COMPLETE.json')['status']=='complete';plan=read(R/'plan.json')
 for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
 features=pd.read_parquet(R/'features.parquet').set_index('application_id');assert b.sha(R/'features.parquet')==read(R/'BUILD_COMPLETE.json')['feature_hash']
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id']);seen=np.zeros(70000,int);pr=np.full(70000,np.nan);checks=[]
 for f in range(5):
  d=R/f'models/fold{f}';done=read(d/'complete.json')
  for n,h in done['files'].items():assert b.sha(d/n)==h
  ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');packet=DATA/f'fold{f}';q=pd.read_parquet(packet/'query.parquet');labels=np.load(packet/'correction_labels.npy').astype(np.int64)
  assert'target'not in q and len(labels)==6000 and q.application_id.tolist()==original.application_id.iloc[np.r_[ix['correction'],ix['validation']]].tolist()
  model=joblib.load(d/'head.joblib');sp=joblib.load(d/'spline.joblib');assert len(model['columns'])==20 and len(sp['columns'])==79
  rawroot=A/f'kumo-coverage-placement-20261021/conditions/fold{f}/control90';assert b.sha(rawroot/'raw_predictions.npz')==read(rawroot/'complete.json')['small_files']['raw_predictions.npz'];raw=np.load(rawroot/'raw_predictions.npz')['prediction']
  prior=A/f'repayment-severity-fe-20261021/fold{f}';means=pd.read_parquet(prior/'features.parquet').set_index('application_id');assert b.sha(prior/'features.parquet')==read(prior/'transformer.json')['feature_hash']
  extras=['support_full_account_months','support_distinct_scored_bureau_reports'];oldcols=[c for c in sp['columns']if c not in extras]
  fixed=A/f'fold-local-priors-20261021/fold{f}';assert b.sha(fixed/'features.parquet')==read(fixed/'complete.json')['feature_table_sha256']
  val=pd.read_parquet(fixed/'features.parquet',columns=['application_id']+oldcols).set_index('application_id').loc[q.application_id.iloc[6000:]].reset_index()
  for c in sp['categorical']:val[c]=val[c].astype(object)
  for t in [1,30]:
   q[f're_late_{t}_posterior']=means.loc[q.application_id,f'matched_beta_late_{t}'].to_numpy(float);val[f're_late_{t}_posterior']=means.loc[val.application_id,f'matched_beta_late_{t}'].to_numpy(float)
  for c in extras:q[c]=features.loc[q.application_id,c].to_numpy(float);val[c]=features.loc[val.application_id,c].to_numpy(float)
  x,o=b.design(model,q.iloc[:6000],raw[:6000]);value,g=b.objective(model['coefficients'],x,o,labels,np.ones(6000),.003);assert abs(value-model['objective'])<1e-12 and max(abs(g))<1e-4
  got=.75*b.corrected_probability(model,q.iloc[6000:],raw[6000:])[0]+.25*b.predict_spline(sp,val);saved=np.load(d/'validation.npz');assert np.array_equal(saved['rows'],ix['validation'])and saved['application_id'].tolist()==val.application_id.tolist();assert np.array_equal(got,saved['prediction'])
  assert sp['resolved_parameters']['C']==.05 and sp['resolved_parameters']['max_iter']==80 and sp['resolved_parameters']['tol']==1e-5 and sp['resolved_parameters']['solver']=='newton-cholesky'
  assert sp['fit_rows']==56000;pr[ix['validation']]=got;seen[ix['validation']]+=1;checks.append(dict(fold=f,models_replayed=2,exact_prediction_replay=True,head_objective_verified=True,original_hyperparameters_verified=True,OOF_rows=14000))
 assert np.all(seen==1);z=np.load(R/'OOF.npz');y=z['target'].astype(np.int64);assert np.array_equal(pr,z['support']);order=np.argsort(pr,kind='stable');v=pr[order];yy=y[order];ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1];n=int(y.sum());ks=float(abs(np.cumsum(yy)[ends]/n-np.cumsum(1-yy)[ends]/(len(y)-n)).max());auc=float((rankdata(pr)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)));reported=read(R/'result.json')['metrics']['support'];assert abs(ks-reported['KS'])<1e-12 and abs(auc-reported['AUC'])<1e-12
 b.dump(dict(status='passed',metrics=dict(KS=ks,AUC=auc),folds=checks,models_replayed=10,OOF_rows=70000,source_sha256=b.sha(Path(__file__)),metric_method='Independent tiedempiricalCDF/ranksumAUC',confirmation_opened=False),R/'INDEPENDENT_AUDIT.json');print(json.dumps(dict(status='passed',KS=ks,AUC=auc,models_replayed=10)),flush=True)
if __name__=='__main__':main()
