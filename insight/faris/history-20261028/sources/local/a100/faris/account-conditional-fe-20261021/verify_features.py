"""Independent scalar conditional-normal reconstruction before classifier fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'));import baseline as b
A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def read(p):return json.loads(p.read_text())
def main():
 done=read(R/'BUILD_COMPLETE.json');assert done['status']=='complete'and done['default_labels_opened']is False
 assert b.sha(R/'source/build.py')==done['source_sha256']and b.sha(R/'source/measurement_model.py')==done['measurement_model_sha256']
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True);apps.decision_datetime=pd.to_datetime(apps.decision_datetime)
 ac=pd.read_csv(RAW/'account_behavior.csv',usecols=['customer_id','month','income_like_credits','outflows'],parse_dates=['month']);assert b.sha(RAW/'account_behavior.csv')==done['source_raw_sha256'];profile=pd.read_csv(RAW/'customer_profiles.csv',usecols=['customer_id','declared_monthly_income']).set_index('customer_id')
 checks=[]
 for f in range(5):
  tx=read(R/f'fold{f}/transformer.json');assert b.sha(R/f'fold{f}/features.parquet')==tx['feature_hash'];ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');assert tx['fit_ids_sha256']==b.ids_hash(original,ix['context'])and tx['default_labels_opened']is False and tx['query_covariates_fit_shared_parameters']is False
  values=pd.read_parquet(R/f'fold{f}/features.parquet');assert values.application_id.tolist()==apps.application_id.tolist()
  # Twenty-four deterministic samples span context, correction, validation andtest.
  rng=np.random.default_rng(20261125+f);rows=np.r_[rng.choice(ix['context'],6,replace=False),rng.choice(ix['correction'],6,replace=False),rng.choice(ix['validation'],6,replace=False),rng.choice(np.arange(70000,100000),6,replace=False)];errors=[]
  for row in rows:
   cid=apps.customer_id.iloc[row];events=ac[ac.customer_id.eq(cid)].sort_values('month');cutoff=apps.decision_datetime.iloc[row];events=events[events.month+pd.offsets.MonthBegin(1)<=cutoff];actual=values.iloc[row]
   if events.empty:
    assert actual[done['replacement_columns']].isna().all();continue
   n=len(events);means={};var={}
   for variable in ['income_like_credits','outflows']:
    params=tx['parameters'][variable];tau=params['between_variance'];sigma=params['measurement_variance'];phi=params['phi'];mu=params['mean'];y=np.log(events[variable].to_numpy(float));positions=np.arange(n)
    cov=tau+sigma*np.power(phi,abs(positions[:,None]-positions[None,:]));cov=cov+np.eye(n)*1e-9;cross=tau+sigma*np.power(phi,n-positions)
    means[variable]=float(mu+cross@np.linalg.solve(cov,y-mu));var[variable]=float(tau+sigma+1e-9-cross@np.linalg.solve(cov,cross))
   inc=means['income_like_credits'];out=means['outflows'];expected={'re_income_geometric_mean':np.exp(inc),'re_income_log_mean':inc,'re_declared_income_log_gap':np.log(profile.loc[cid,'declared_monthly_income'])-inc,'re_outflow_income_geometric_ratio':np.exp(out-inc),'re_income_log_measurement_sd':np.sqrt(var['income_like_credits'])}
   for key,v in expected.items():assert np.isclose(actual[key],v,rtol=1e-11,atol=1e-10),(f,row,key,actual[key],v)
   errors.append(max(abs(actual[k]-v)/max(abs(v),1)for k,v in expected.items()))
  checks.append(dict(fold=f,sample_applicants=24,maximum_scaled_error=max(errors),fitted_context_ID_hash_verified=True,query_covariates_not_used_to_fit_prior=True))
 b.dump(dict(status='passed',checks=checks,sampled_applicants=120,columns=5,default_labels_opened=False,source_sha256=b.sha(Path(__file__)),scope='Scalar conditionalGaussianmatrixreconstructionwithserializedparams acrossall5folds andallinputroles; codefit-scope/IDhash inspected, notproof of future defaultgeneralization.'),R/'INDEPENDENT_FEATURE_AUDIT.json');print('Independent feature reconstruction passed all5folds.',flush=True)
if __name__=='__main__':main()
