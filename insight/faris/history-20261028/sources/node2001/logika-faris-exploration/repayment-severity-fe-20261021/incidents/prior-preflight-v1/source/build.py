"""Target-free joint severity feature; independent matched-maturity control."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.special import gammaln,digamma,betaln
from scipy.optimize import minimize
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def dm_loss(theta,counts):
 alpha=np.exp(theta);total=alpha.sum();n=counts.sum(axis=1)
 value=gammaln(total)-gammaln(total+n)+(gammaln(counts+alpha)-gammaln(alpha)).sum(axis=1)
 grad=(digamma(total)-digamma(total+n))[:,None]+digamma(counts+alpha)-digamma(alpha)
 return -float(value.mean()),-grad.mean(axis=0)*alpha
def beta_fit(k,n):
 good=n>0;k=k[good];n=n[good];m=float(k.sum()/n.sum())
 def fun(theta):
  aa,bb=np.exp(theta);v=betaln(k+aa,n-k+bb)-betaln(aa,bb)
  ga=digamma(k+aa)-digamma(n+aa+bb)-digamma(aa)+digamma(aa+bb)
  gb=digamma(n-k+bb)-digamma(n+aa+bb)-digamma(bb)+digamma(aa+bb)
  return -float(v.mean()),-np.array([ga.mean()*aa,gb.mean()*bb])
 result=minimize(fun,np.log([m*12,(1-m)*12]),jac=True,method='L-BFGS-B',bounds=[(-7,10)]*2,options=dict(maxiter=200,ftol=1e-11,gtol=1e-7))
 assert result.success,result.message
 return dict(alpha=float(np.exp(result.x[0])),beta=float(np.exp(result.x[1])),objective=float(result.fun),gradient_max_abs=float(abs(result.jac).max()),iterations=int(result.nit))
def main():
 assert not(R/'BUILD_COMPLETE.json').exists();assert time.time()<1792585871.823748-10800
 start=time.time();apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True)
 assert len(apps)==100000 and apps.customer_id.is_unique;apps.decision_datetime=pd.to_datetime(apps.decision_datetime)
 raw=RAW/'repayment_history.csv';d=pd.read_csv(raw,usecols=['customer_id','installment_id','due_date','max_days_past_due'],parse_dates=['due_date']).merge(apps,on='customer_id',validate='many_to_one')
 assert not d.duplicated(['customer_id','installment_id']).any();age=(d.decision_datetime-d.due_date).dt.total_seconds()/86400;assert (age>0).all()
 assert d.max_days_past_due.ge(0).all();eligible=d[age>=30].copy();eligible['category']=np.searchsorted([1,7,30],eligible.max_days_past_due,side='right')
 # Exact integer counts preserve the category nesting and are row-local.
 counts=eligible.groupby(['customer_id','category']).size().unstack(fill_value=0).reindex(index=apps.customer_id,columns=range(4),fill_value=0).astype(np.int64)
 assert int(counts.to_numpy().sum())==len(eligible)
 assert np.array_equal(counts.sum(axis=1).to_numpy(),eligible.groupby('customer_id').size().reindex(apps.customer_id,fill_value=0).to_numpy())
 # Independent threshold reconstruction across all100k applicants.
 for threshold,bins in [(1,[1,2,3]),(7,[2,3]),(30,[3])]:
  direct=eligible[eligible.max_days_past_due>=threshold].groupby('customer_id').size().reindex(apps.customer_id,fill_value=0)
  assert np.array_equal(counts[bins].sum(axis=1).to_numpy(),direct.to_numpy())
 out=apps[['application_id','customer_id']].copy();out[[f'count{i}'for i in range(4)]]=counts.to_numpy();out.to_parquet(R/'severity_counts.parquet',index=False)
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);assert original.application_id.tolist()==apps.application_id.iloc[:70000].tolist()
 summaries=[];allc=counts.to_numpy(float);n=allc.sum(axis=1)
 for fold in range(5):
  assert time.time()<1792585871.823748-10800;dest=R/f'fold{fold}';dest.mkdir()
  ix=np.load(INPUT/f'artifacts/fold{fold}/indices.npz');ctx=ix['context'];train=allc[ctx];train=train[train.sum(axis=1)>0]
  initial=(train.sum(axis=0)+1)/(train.sum()+4)*12
  # Independent centered numerical gradient check before optimization.
  theta=np.log(initial);value,gradient=dm_loss(theta,train);eps=1e-5;num=[]
  for j in range(4):
   step=np.zeros(4);step[j]=eps;num.append((dm_loss(theta+step,train)[0]-dm_loss(theta-step,train)[0])/(2*eps))
  assert np.max(abs(np.array(num)-gradient))<1e-5
  result=minimize(dm_loss,theta,args=(train,),jac=True,method='L-BFGS-B',bounds=[(-7,10)]*4,options=dict(maxiter=200,ftol=1e-11,gtol=1e-7))
  assert result.success and max(abs(result.jac))<1e-4,(result.message,result.jac)
  alpha=np.exp(result.x);posterior=(allc+alpha)/(n[:,None]+alpha.sum());assert np.allclose(posterior.sum(axis=1),1)
  beta={}
  features=apps[['application_id','customer_id']].copy()
  for threshold,bins in [(1,[1,2,3]),(30,[3])]:
   k=allc[:,bins].sum(axis=1);pars=beta_fit(k[ctx],n[ctx]);beta[str(threshold)]=pars
   features[f'joint_late_{threshold}']=posterior[:,bins].sum(axis=1)
   features[f'matched_beta_late_{threshold}']=(k+pars['alpha'])/(n+pars['alpha']+pars['beta'])
  features.to_parquet(dest/'features.parquet',index=False)
  params=dict(dirichlet_alpha=alpha.tolist(),objective=float(result.fun),gradient_max_abs=float(abs(result.jac).max()),iterations=int(result.nit),success=bool(result.success),beta=beta,eligible_training_customers=len(train),fit_rows=50000,fit_ids_sha256=b.ids_hash(original,ctx),feature_hash=b.sha(dest/'features.parquet'),source_count_hash=b.sha(R/'severity_counts.parquet'),feature_fit_default_labels_opened=False,query_covariates_fit_shared_prior=False)
  b.dump(params,dest/'transformer.json');summaries.append(params);print(json.dumps(dict(stage='severity_feature_fold_complete',fold=fold,alpha=alpha.tolist())),flush=True)
 b.dump(dict(status='complete',rows=100000,eligible_installments=len(eligible),threshold_counts_independently_verified=True,target_labels_opened=False,maturity_days=30,categories=['0','1-6','7-29','30+'],fit_context_only=True,folds=summaries,raw_source_sha256=b.sha(raw),build_source_sha256=b.sha(Path(__file__)),seconds=time.time()-start),R/'BUILD_COMPLETE.json')
if __name__=='__main__':main()
