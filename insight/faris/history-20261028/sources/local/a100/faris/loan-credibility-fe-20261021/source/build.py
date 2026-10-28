import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time,json,gc
import numpy as np,pandas as pd
from scipy.special import roots_jacobi,betaln,logsumexp
from scipy.sparse import csr_matrix
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input';PRIOR=A/'repayment-severity-fe-20261021'
def check():assert time.time()<1792585871.823748-10800 and not(R/'STOP').exists()
def moment(loan,fit_ids):
 x=loan[loan.customer_id.isin(fit_ids)].copy();c=x.groupby('customer_id')[['k','n']].sum();mean=x.customer_id.map(c.k/c.n);total=x.customer_id.map(c.n)
 N=int(x.n.sum());L=len(x);I=len(c);within=float((x.k-x.k*x.k/x.n).sum());between=float((x.n*(x.k/x.n-mean)**2).sum());sigma=within/max(N-L,1);denom=float(N-(x.n*x.n/total).sum());tau=(between-sigma*(L-I))/denom if denom>0 else 0.;rho=max(tau,0)/(max(tau,0)+sigma)if sigma+max(tau,0)>0 else 0.
 return dict(N=N,loans=L,borrowers=I,within_variance=sigma,unconstrained_loan_variance=tau,rho=rho,kappa=(1-rho)/rho if rho>0 else None)
def posterior(matrix,patterns,alpha,beta,kappa,nodes=256):
 xx,weights=roots_jacobi(nodes,beta-1,alpha-1);p=(xx+1)/2;weights=weights/weights.sum();n=patterns[:,0,None];k=patterns[:,1,None]
 if kappa is None:ll=k*np.log(p)[None,:]+(n-k)*np.log1p(-p)[None,:]
 else:ll=betaln(k+kappa*p,(n-k)+kappa*(1-p))-betaln(kappa*p,kappa*(1-p))
 lp=np.asarray(matrix@ll)+np.log(weights)[None,:];lp-=logsumexp(lp,axis=1)[:,None];prob=np.exp(lp);mean=prob@p;variance=np.maximum(prob@(p*p)-mean*mean,0)
 return mean,np.sqrt(variance)
def main():
 check();assert not(R/'BUILD_COMPLETE.json').exists();start=time.time()
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True);apps.decision_datetime=pd.to_datetime(apps.decision_datetime);assert apps.customer_id.is_unique;apps['row']=np.arange(len(apps))
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);assert apps.application_id.iloc[:70000].tolist()==original.application_id.tolist()
 rp=pd.read_csv(RAW/'repayment_history.csv',usecols=['customer_id','loan_id','installment_id','due_date','max_days_past_due'],parse_dates=['due_date']).merge(apps,on='customer_id',validate='many_to_one')
 assert not rp.duplicated(['customer_id','installment_id']).any();age=(rp.decision_datetime-rp.due_date).dt.total_seconds()/86400;assert(age>0).all();rp=rp[age>=30].copy()
 output={f:apps[['application_id','customer_id']].copy()for f in range(5)};audit=[]
 for t in [1,30]:
  check();rp['late']=rp.max_days_past_due.ge(t).astype(np.int64)
  loan=rp.groupby(['customer_id','row','loan_id']).late.agg(k='sum',n='size').reset_index();loan.to_parquet(R/f'loan_counts_{t}.parquet',index=False)
  maximum=int(loan.groupby('row').n.sum().max());assert maximum<=254,maximum
  # Independent borrower/threshold count reconciliation with preceding feature study.
  prior_counts=pd.read_parquet(PRIOR/'severity_counts.parquet');assert prior_counts.application_id.tolist()==apps.application_id.tolist()
  expected_n=prior_counts[[f'count{i}'for i in range(4)]].sum(axis=1).to_numpy();expected_k=prior_counts[[f'count{i}'for i in ([1,2,3]if t==1 else[3])]].sum(axis=1).to_numpy()
  assert np.array_equal(loan.groupby('row').n.sum().reindex(range(len(apps)),fill_value=0).to_numpy(),expected_n)
  assert np.array_equal(loan.groupby('row').k.sum().reindex(range(len(apps)),fill_value=0).to_numpy(),expected_k)
  codes,unique=pd.factorize(pd.MultiIndex.from_frame(loan[['n','k']]),sort=True);patterns=np.array(unique.tolist(),dtype=np.int64)
  matrix=csr_matrix((np.ones(len(loan)),(loan.row.to_numpy(),codes)),shape=(len(apps),len(patterns)));matrix.sum_duplicates()
  for f in range(5):
   check();ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');fit_ids=set(original.customer_id.iloc[ix['context']]);params=moment(loan,fit_ids);prior=json.loads((PRIOR/f'fold{f}/transformer.json').read_text())['beta'][str(t)];aa,bb=prior['alpha'],prior['beta']
   assert params['rho']>0,'Persistent loan effect not supported; do not admit this feature.'
   iid,_=posterior(matrix,patterns,aa,bb,None,128);analytical=(expected_k+aa)/(expected_n+aa+bb);iid_error=float(abs(iid-analytical).max());assert iid_error<1e-9,iid_error;del iid
   mean,sd=posterior(matrix,patterns,aa,bb,params['kappa'],256);other,_=posterior(matrix,patterns,aa,bb,params['kappa'],128);err=float(abs(mean-other).max());assert err<1e-6,err
   output[f][f'cluster_late_{t}']=mean;output[f][f'cluster_late_{t}_sd']=sd
   audit.append(dict(fold=f,threshold=t,loan_moments=params,prior=prior,prior_source_hash=b.sha(PRIOR/f'fold{f}/transformer.json'),fit_ids_sha256=b.ids_hash(original,ix['context']),quadrature_128_vs_256_max_error=err,iid_analytical_max_error=iid_error,max_installments_per_borrower=maximum,default_labels_opened=False));del mean,other,sd;gc.collect()
 for f,frame in output.items():
  d=R/f'fold{f}';d.mkdir();frame.to_parquet(d/'features.parquet',index=False);b.dump(dict(fold=f,feature_hash=b.sha(d/'features.parquet'),thresholds=[x for x in audit if x['fold']==f],query_covariates_fit_shared_parameters=False),d/'transformer.json')
 b.dump(dict(status='complete',default_labels_opened=False,rows=100000,feature_inputs=2,thresholds=audit,source_sha256=b.sha(Path(__file__)),raw_sha256=b.sha(RAW/'repayment_history.csv'),seconds=time.time()-start),R/'BUILD_COMPLETE.json');print(json.dumps(dict(status='complete',seconds=time.time()-start,diagnostics=audit)),flush=True)
if __name__=='__main__':main()
