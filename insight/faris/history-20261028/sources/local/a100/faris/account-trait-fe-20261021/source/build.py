import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np,pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input';PARENT=A/'account-conditional-fe-20261021'
REPLACE=['re_income_geometric_mean','re_income_log_mean','re_declared_income_log_gap','re_outflow_income_geometric_ratio','re_income_log_measurement_sd']
def read(p):return json.loads(p.read_text())
def main():
 assert not(R/'BUILD_COMPLETE.json').exists()and time.time()<1792585871.823748-10800;start=time.time();assert read(PARENT/'FINAL_STATUS.json')['status']=='passed'
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True);apps.decision_datetime=pd.to_datetime(apps.decision_datetime);assert apps.customer_id.is_unique
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);assert original.application_id.tolist()==apps.application_id.iloc[:70000].tolist();profiles=pd.read_csv(RAW/'customer_profiles.csv',usecols=['customer_id','declared_monthly_income']).set_index('customer_id');declared=profiles.declared_monthly_income.reindex(apps.customer_id).to_numpy(float);assert np.all(declared>0)
 ac=pd.read_csv(RAW/'account_behavior.csv',usecols=['customer_id','month','income_like_credits','outflows'],parse_dates=['month']).merge(apps,on='customer_id',validate='many_to_one');assert (ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime).all()and not ac.duplicated(['customer_id','month']).any();ac=ac.sort_values(['customer_id','month']);serial=ac.month.dt.year*12+ac.month.dt.month;assert serial.groupby(ac.customer_id).diff().dropna().eq(1).all();assert ac[['income_like_credits','outflows']].gt(0).all().all()
 groups={};count=ac.groupby('customer_id').size()
 for n,customers in count.groupby(count).groups.items():
  part=ac[ac.customer_id.isin(customers)];ids=part.customer_id.to_numpy().reshape(len(customers),int(n))[:,0];groups[int(n)]=dict(ids=ids,income_like_credits=np.log(part.income_like_credits.to_numpy(float)).reshape(len(customers),int(n)),outflows=np.log(part.outflows.to_numpy(float)).reshape(len(customers),int(n)))
 receipts=[]
 for f in range(5):
  assert time.time()<1792585871.823748-10800;source=PARENT/f'fold{f}/transformer.json';old=read(source);ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');assert old['fit_ids_sha256']==b.ids_hash(original,ix['context'])and old['query_covariates_fit_shared_parameters']is False
  estimates={};uncertainties={}
  for variable in ['income_like_credits','outflows']:
   pp=old['parameters'][variable];tau=pp['between_variance'];sigma=pp['measurement_variance'];phi=pp['phi'];mu=pp['mean'];values=[];sd=[];ids=[]
   for n,g in groups.items():
    pos=np.arange(n);cov=tau+sigma*phi**abs(pos[:,None]-pos[None,:])+np.eye(n)*1e-9;cross=np.full(n,tau);weight=np.linalg.solve(cov,cross);posterior=mu+(g[variable]-mu)@weight;variance=tau-float(cross@weight);assert 0<variance<tau
    values.append(posterior);sd.append(np.full(len(posterior),np.sqrt(variance)));ids.append(g['ids'])
    # Independent precision-form posterior under the same covariance model.
    noise=sigma*phi**abs(pos[:,None]-pos[None,:])+np.eye(n)*1e-9;one=np.ones(n);ivone=np.linalg.solve(noise,one);v2=1/(1/tau+one@ivone);direct=v2*(mu/tau+g[variable]@ivone);assert abs(v2-variance)<1e-8 and np.max(abs(direct-posterior))<1e-9
   idx=np.concatenate(ids);estimates[variable]=pd.Series(np.concatenate(values),index=idx).reindex(apps.customer_id).to_numpy();uncertainties[variable]=pd.Series(np.concatenate(sd),index=idx).reindex(apps.customer_id).to_numpy()
  inc=estimates['income_like_credits'];outflow=estimates['outflows'];features=apps[['application_id','customer_id']].copy();features[REPLACE[0]]=np.exp(inc);features[REPLACE[1]]=inc;features[REPLACE[2]]=np.log(declared)-inc;features[REPLACE[3]]=np.exp(outflow-inc);features[REPLACE[4]]=uncertainties['income_like_credits'];assert np.isfinite(features[REPLACE].dropna().to_numpy()).all()
  dest=R/f'fold{f}';dest.mkdir();features.to_parquet(dest/'features.parquet',index=False);tx=dict(fold=f,parameters=old['parameters'],fit_rows=50000,fit_ids_sha256=old['fit_ids_sha256'],feature_hash=b.sha(dest/'features.parquet'),source_measurement_parameter_hash=b.sha(source),default_labels_opened=False,query_covariates_fit_shared_parameters=False,new_shared_parameters_fitted=False,precision_form_posterior_independently_verified=True,estimand='persistentGaussianrandomintercept')
  b.dump(tx,dest/'transformer.json');receipts.append(tx);print(json.dumps(dict(stage='feature_fold_complete',fold=f)),flush=True)
 b.dump(dict(status='complete',rows=100000,replacement_columns=REPLACE,default_labels_opened=False,new_shared_parameters_fitted=False,folds=receipts,source_raw_sha256=b.sha(RAW/'account_behavior.csv'),measurement_model_sha256=b.sha(PARENT/'source/measurement_model.py'),source_sha256=b.sha(Path(__file__)),seconds=time.time()-start),R/'BUILD_COMPLETE.json')
if __name__=='__main__':main()
