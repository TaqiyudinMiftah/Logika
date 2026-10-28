import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import multivariate_normal
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def covariance(n,tau,sigma,phi):
 x=np.arange(n);return tau*np.ones((n,n))+sigma*phi**abs(x[:,None]-x[None,:])+np.eye(n)*1e-9
def fit(groups,mu,correlated):
 stats=[]
 for n,y in groups.items():
  centered=y-mu;stats.append((n,len(y),centered.T@centered))
 def unpack(theta):return np.exp(theta[0]),np.exp(theta[1]),float(expit(theta[2]))if correlated else 0.
 def objective(theta):
  tau,sigma,phi=unpack(theta);value=0.;N=0
  for n,count,scatter in stats:
   cov=covariance(n,tau,sigma,phi);sign,logdet=np.linalg.slogdet(cov)
   if sign<=0:return 1e20
   value+=count*(n*np.log(2*np.pi)+logdet)+float(np.trace(np.linalg.solve(cov,scatter)));N+=count
  return .5*value/N
 init=np.array([np.log(1.),np.log(.15),0.])if correlated else np.log([1.,.15]);bounds=[(-12,4),(-12,4)]+([(-8,4)]if correlated else[])
 result=minimize(objective,init,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=100,ftol=1e-11,gtol=1e-6));assert result.success,result.message
 tau,sigma,phi=unpack(result.x)
 # Independent individual log-density check on a fixed subset at the optimum.
 n=sorted(groups)[0];example=groups[n][:16];cov=covariance(n,tau,sigma,phi);direct=-float(multivariate_normal.logpdf(example,mean=np.full(n,mu),cov=cov).sum());center=example-mu;_,ld=np.linalg.slogdet(cov);grouped=.5*(len(example)*(n*np.log(2*np.pi)+ld)+np.trace(np.linalg.solve(cov,center.T@center)));assert abs(direct-grouped)<1e-8
 return dict(mean=mu,between_variance=float(tau),measurement_variance=float(sigma),phi=float(phi),objective=float(result.fun),iterations=int(result.nit),success=bool(result.success),gradient_max_abs=float(abs(result.jac).max()),individual_likelihood_check_max_error=float(abs(direct-grouped)))
def forecast(prefix,params):
 n=prefix.shape[1];tau=params['between_variance'];sigma=params['measurement_variance'];phi=params['phi'];mu=params['mean'];cov=covariance(n,tau,sigma,phi);cross=tau+sigma*phi**(np.arange(n,n+3)[:,None]-np.arange(n)[None,:]);weights=np.linalg.solve(cov,cross.T).T
 result=mu+(prefix-mu)@weights.T
 direct=mu+cross@np.linalg.solve(cov,prefix[0]-mu);assert abs(direct-result[0]).max()<1e-10
 return result
def main():
 assert not(R/'RESULT.json').exists()and time.time()<1792585871.823748-10800;start=time.time()
 apps=pd.read_csv(RAW/'train_applications.csv',usecols=['application_id','customer_id','decision_datetime']);original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);apps=apps.set_index('application_id').loc[original.application_id].reset_index();apps.decision_datetime=pd.to_datetime(apps.decision_datetime);ix=np.load(INPUT/'artifacts/fold0/indices.npz');context=apps.iloc[ix['context']];assert len(context)==50000
 raw=RAW/'account_behavior.csv';ac=pd.read_csv(raw,usecols=['customer_id','month','income_like_credits','outflows'],parse_dates=['month']).merge(context,on='customer_id',validate='many_to_one');assert (ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime).all()and not ac.duplicated(['customer_id','month']).any();ac=ac.sort_values(['customer_id','month']);serial=ac.month.dt.year*12+ac.month.dt.month;delta=serial.groupby(ac.customer_id).diff().dropna();assert delta.eq(1).all()
 count=ac.groupby('customer_id').size();assert count.min()>=8;results={}
 for variable in ['income_like_credits','outflows']:
  assert time.time()<1792585871.823748-10800;assert ac[variable].gt(0).all();groups={};held={};ids={}
  for n,customers in count.groupby(count).groups.items():
   part=ac[ac.customer_id.isin(customers)];values=np.log(part[variable].to_numpy(float)).reshape(len(customers),int(n));ordered=part.customer_id.to_numpy().reshape(len(customers),int(n))[:,0];assert len(np.unique(ordered))==len(customers)
   groups[int(n)-3]=values[:,:-3];held[int(n)-3]=values[:,-3:];ids[int(n)-3]=ordered
  mu=float(np.concatenate([v.mean(axis=1)for v in groups.values()]).mean());params={name:fit(groups,mu,name=='AR1')for name in ['iid','AR1']}
  predictions={name:[]for name in ['iid','AR1','prefix_mean']};truth=[];customers=[]
  for n,y in sorted(groups.items()):
   truth.append(held[n]);customers.append(ids[n]);predictions['prefix_mean'].append(np.repeat(y.mean(axis=1)[:,None],3,axis=1))
   for name in ['iid','AR1']:predictions[name].append(forecast(y,params[name]))
  truth=np.concatenate(truth);customers=np.concatenate(customers);predictions={k:np.concatenate(v)for k,v in predictions.items()};errors={k:((v-truth)**2).mean(axis=1)for k,v in predictions.items()};delta=errors['AR1']-errors['iid'];rng=np.random.default_rng(20261123);bootstrap=[float(delta[rng.integers(0,len(delta),len(delta))].mean())for _ in range(1000)]
  diagnostics={k:dict(MSE=float(v.mean()),RMSE=float(np.sqrt(v.mean())),first_month_MSE=float(((predictions[k][:,0]-truth[:,0])**2).mean()),signed_log_error=float((predictions[k]-truth).mean()))for k,v in errors.items()}
  np.savez_compressed(R/(variable+'_forecasts.npz'),customer_id=customers.astype(str),truth=truth,**predictions)
  results[variable]=dict(parameters=params,diagnostics=diagnostics,customers=len(truth),AR1_minus_iid_MSE=float(delta.mean()),paired_customer_interval95=np.quantile(bootstrap,[.0125,.9875]).tolist(),forecast_prefix_lengths=sorted(groups),heldout_months_per_customer=3,default_labels_opened=False,scope='Simultaneous97.5percentintervalforeachof2variables,conditionalforecasts; withhold3monthsandneverfitsharedparametersonthem. Not default-model generalization.')
  b.dump(results[variable],R/(variable+'_result.json'));print(json.dumps(dict(stage='variable_complete',variable=variable,diagnostics=diagnostics,phi=params['AR1']['phi'])),flush=True)
 b.dump(dict(status='complete',context_rows=50000,context_ids_sha256=b.ids_hash(original,ix['context']),default_labels_opened=False,outer_and_correction_histories_used=False,source_raw_sha256=b.sha(raw),source_sha256=b.sha(Path(__file__)),results=results,seconds=time.time()-start,classifier_training_admitted=False),R/'RESULT.json')
if __name__=='__main__':main()
