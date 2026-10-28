"""Independently reconstruct both full5fold feature conditions and metrics."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'));import baseline as b
from build import EXTRA
A=R.parent;INPUT=A/'ranking-pipeline-20261020/input';DATA=A/'kumo-temperature-adaptation-20261021/full_data'
def read(p):return json.loads(p.read_text())
def main():
 assert read(R/'TRAINING_COMPLETE.json')['models']==10;plan=read(R/'plan.json');audit=read(R/'BUILD_COMPLETE.json')
 for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id']);z=np.load(R/'OOF.npz');y=z['target'].astype(np.int64);fold=z['fold'];reported=read(R/'result.json');checks=[];metrics={}
 for kind in ['occasion']:
  out=np.full(70000,np.nan);seen=np.zeros(70000,int)
  for f in range(5):
   features={kind:pd.read_parquet(R/f'fold{f}/features.parquet').set_index('application_id')};assert b.sha(R/f'fold{f}/features.parquet')==audit['folds'][f]['feature_sha256']
   root=R/f'models/{kind}/fold{f}'
   for n,h in read(root/'complete.json')['files'].items():assert b.sha(root/n)==h
   ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');q=pd.read_parquet(DATA/f'fold{f}/query.parquet');labels=np.load(DATA/f'fold{f}/correction_labels.npy').astype(np.int64);assert'target'not in q and len(labels)==6000
   assert q.application_id.tolist()==original.application_id.iloc[np.r_[ix['correction'],ix['validation']]].tolist()
   model=joblib.load(root/'head.joblib');sp=joblib.load(root/'spline.joblib');assert len(model['columns'])==18 and len(sp['columns'])==77;assert all(c in model['columns']and c in sp['columns']for c in EXTRA)
   rawroot=A/f'kumo-coverage-placement-20261021/conditions/fold{f}/control90';assert b.sha(rawroot/'raw_predictions.npz')==read(rawroot/'complete.json')['small_files']['raw_predictions.npz'];raw=np.load(rawroot/'raw_predictions.npz')['prediction']
   prior=A/f'repayment-severity-fe-20261021/fold{f}';assert b.sha(prior/'features.parquet')==read(prior/'transformer.json')['feature_hash'];means=pd.read_parquet(prior/'features.parquet').set_index('application_id')
   fixed=A/f'fold-local-priors-20261021/fold{f}';assert b.sha(fixed/'features.parquet')==read(fixed/'complete.json')['feature_table_sha256'];oldcols=sp['columns'];val=pd.read_parquet(fixed/'features.parquet',columns=['application_id']+oldcols).set_index('application_id').loc[q.application_id.iloc[6000:]].reset_index()
   for c in sp['categorical']:val[c]=val[c].astype(object)
   for t in [1,30]:q[f're_late_{t}_posterior']=means.loc[q.application_id,f'matched_beta_late_{t}'].to_numpy(float);val[f're_late_{t}_posterior']=means.loc[val.application_id,f'matched_beta_late_{t}'].to_numpy(float)
   for c in EXTRA:q[c]=features[kind].loc[q.application_id,c].to_numpy(float);val[c]=features[kind].loc[val.application_id,c].to_numpy(float)
   x,o=b.design(model,q.iloc[:6000],raw[:6000]);value,g=b.objective(model['coefficients'],x,o,labels,np.ones(6000),.003);assert abs(value-model['objective'])<1e-12 and max(abs(g))<1e-4
   assert model['spec']['correction_strength']==.5 and sp['resolved_parameters']['C']==.05 and sp['resolved_parameters']['solver']=='newton-cholesky'and sp['resolved_parameters']['max_iter']==80 and sp['resolved_parameters']['tol']==1e-5
   prediction=.75*b.corrected_probability(model,q.iloc[6000:],raw[6000:])[0]+.25*b.predict_spline(sp,val);saved=np.load(root/'validation.npz');assert np.array_equal(saved['rows'],ix['validation'])and saved['application_id'].tolist()==val.application_id.tolist();assert np.array_equal(prediction,saved['prediction']);out[ix['validation']]=prediction;seen[ix['validation']]+=1;checks.append(dict(fold=f,variant=kind,models_replayed=2,exact_replay=True,head_objective_verified=True,original_hyperparameters_verified=True))
  assert np.all(seen==1)and np.array_equal(out,z[kind]);order=np.argsort(out,kind='stable');v=out[order];yy=y[order];ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1];n=int(y.sum());ks=float(abs(np.cumsum(yy)[ends]/n-np.cumsum(1-yy)[ends]/(len(y)-n)).max());auc=float((rankdata(out)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)));assert abs(ks-reported['metrics'][kind]['KS'])<1e-12 and abs(auc-reported['metrics'][kind]['AUC'])<1e-12;metrics[kind]=dict(KS=ks,AUC=auc)
 b.dump(dict(status='passed',checks=checks,models_replayed=10,OOF_rows_per_condition=70000,metrics=metrics,source_sha256=b.sha(Path(__file__)),confirmation_opened=False,qualification='Independentfullmodel/feature/objectiveandCDF-rankmetricreconstruction. Doesnotremovehistoricalselectionorretraininguncertainty.'),R/'INDEPENDENT_AUDIT.json');print(json.dumps(dict(status='passed',metrics=metrics,models_replayed=10)),flush=True)
if __name__=='__main__':main()
