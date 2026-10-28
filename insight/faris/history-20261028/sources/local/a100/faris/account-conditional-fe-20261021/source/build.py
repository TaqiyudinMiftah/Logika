import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np,pandas as pd
import baseline as b
from measurement_model import fit,covariance,forecast
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
REPLACE=['re_income_geometric_mean','re_income_log_mean','re_declared_income_log_gap','re_outflow_income_geometric_ratio','re_income_log_measurement_sd']
def main():
 assert not(R/'BUILD_COMPLETE.json').exists()and time.time()<1792585871.823748-10800;start=time.time()
 audit=json.loads((A/'account-measurement-audit-20261021/RESULT.json').read_text());assert audit['status']=='complete'and audit['default_labels_opened']is False
 assert all(v['paired_customer_interval95'][1]<0 for v in audit['results'].values())
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True);apps.decision_datetime=pd.to_datetime(apps.decision_datetime);assert apps.customer_id.is_unique
 profiles=pd.read_csv(RAW/'customer_profiles.csv',usecols=['customer_id','declared_monthly_income']).set_index('customer_id');declared=profiles.declared_monthly_income.reindex(apps.customer_id).to_numpy(float);assert np.all(declared>0)
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);assert original.application_id.tolist()==apps.application_id.iloc[:70000].tolist()
 ac=pd.read_csv(RAW/'account_behavior.csv',usecols=['customer_id','month','income_like_credits','outflows'],parse_dates=['month']).merge(apps,on='customer_id',validate='many_to_one');assert (ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime).all()and not ac.duplicated(['customer_id','month']).any();ac=ac.sort_values(['customer_id','month']);month=ac.month.dt.year*12+ac.month.dt.month;assert month.groupby(ac.customer_id).diff().dropna().eq(1).all();count=ac.groupby('customer_id').size();groups={}
 for n,customers in count.groupby(count).groups.items():
  part=ac[ac.customer_id.isin(customers)];ids=part.customer_id.to_numpy().reshape(len(customers),int(n))[:,0];groups[int(n)]=dict(ids=ids,income_like_credits=np.log(part.income_like_credits.to_numpy(float)).reshape(len(customers),int(n)),outflows=np.log(part.outflows.to_numpy(float)).reshape(len(customers),int(n)))
 receipts=[]
 for fold in range(5):
  assert time.time()<1792585871.823748-10800;ix=np.load(INPUT/f'artifacts/fold{fold}/indices.npz');fit_ids=set(original.customer_id.iloc[ix['context']]);params={};estimates={};uncertainty={}
  for variable in ['income_like_credits','outflows']:
   train={n:g[variable][np.isin(g['ids'],list(fit_ids))]for n,g in groups.items()};train={n:y for n,y in train.items()if len(y)};mu=float(np.concatenate([y.mean(axis=1)for y in train.values()]).mean());pp=fit(train,mu,True);params[variable]=pp;values=[];sds=[];ids=[]
   for n,g in groups.items():
    pred=forecast(g[variable],pp)[:,0];tau=pp['between_variance'];sigma=pp['measurement_variance'];phi=pp['phi'];cross=tau+sigma*phi**(n-np.arange(n));cov=covariance(n,tau,sigma,phi);variance=tau+sigma+1e-9-float(cross@np.linalg.solve(cov,cross));assert variance>0
    values.append(pred);sds.append(np.full(len(pred),np.sqrt(variance)));ids.append(g['ids'])
   order=np.concatenate(ids);estimates[variable]=pd.Series(np.concatenate(values),index=order).reindex(apps.customer_id).to_numpy();uncertainty[variable]=pd.Series(np.concatenate(sds),index=order).reindex(apps.customer_id).to_numpy()
  inc=estimates['income_like_credits'];outflow=estimates['outflows'];features=apps[['application_id','customer_id']].copy();features[REPLACE[0]]=np.exp(inc);features[REPLACE[1]]=inc;features[REPLACE[2]]=np.log(declared)-inc;features[REPLACE[3]]=np.exp(outflow-inc);features[REPLACE[4]]=uncertainty['income_like_credits'];assert np.isfinite(features[REPLACE].dropna().to_numpy()).all()
  dest=R/f'fold{fold}';dest.mkdir();features.to_parquet(dest/'features.parquet',index=False)
  tx=dict(fold=fold,parameters=params,fit_rows=50000,fit_account_customers=int(sum(len(g['ids'][np.isin(g['ids'],list(fit_ids))])for g in groups.values())),fit_ids_sha256=b.ids_hash(original,ix['context']),feature_hash=b.sha(dest/'features.parquet'),default_labels_opened=False,query_covariates_fit_shared_parameters=False,raw_observation_audit_hash=b.sha(A/'account-measurement-audit-20261021/RESULT.json'))
  b.dump(tx,dest/'transformer.json');receipts.append(tx);print(json.dumps(dict(stage='feature_fold_complete',fold=fold,phi={k:v['phi']for k,v in params.items()})),flush=True)
 b.dump(dict(status='complete',rows=100000,replacement_columns=REPLACE,default_labels_opened=False,folds=receipts,source_raw_sha256=b.sha(RAW/'account_behavior.csv'),measurement_model_sha256=b.sha(R/'source/measurement_model.py'),source_sha256=b.sha(Path(__file__)),seconds=time.time()-start),R/'BUILD_COMPLETE.json')
if __name__=='__main__':main()
