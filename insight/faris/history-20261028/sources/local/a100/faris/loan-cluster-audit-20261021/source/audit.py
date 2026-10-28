"""Context-only nested ANOVA diagnostic; never reads applicant default labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,pandas as pd,json,time
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
start=time.time();assert not(R/'RESULT.json').exists();assert time.time()<1792585871.823748-10800
apps=pd.read_csv(RAW/'train_applications.csv',usecols=['application_id','customer_id','decision_datetime']);apps.decision_datetime=pd.to_datetime(apps.decision_datetime)
original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);assert apps.application_id.is_unique
apps=apps.set_index('application_id').loc[original.application_id].reset_index();assert apps.customer_id.tolist()==original.customer_id.tolist()
ix=np.load(INPUT/'artifacts/fold0/indices.npz');context=apps.iloc[ix['context']].copy();assert len(context)==50000
rp=pd.read_csv(RAW/'repayment_history.csv',usecols=['customer_id','loan_id','installment_id','due_date','max_days_past_due'],parse_dates=['due_date']).merge(context,on='customer_id',validate='many_to_one')
age=(rp.decision_datetime-rp.due_date).dt.total_seconds()/86400;assert(age>0).all()
timing=dict(rows=len(rp),max_DPD_exceeds_elapsed_days=int(rp.max_days_past_due.gt(age+1).sum()),DPD_1_to_6_rows=int(rp.max_days_past_due.between(1,6).sum()))
rp=rp[age>=30].copy();out={}
for threshold in [1,30]:
 rp['late']=rp.max_days_past_due.ge(threshold).astype(float)
 loan=rp.groupby(['customer_id','loan_id']).late.agg(['sum','count','mean']);customer=rp.groupby('customer_id').late.agg(['sum','count','mean'])
 cm=loan.index.get_level_values('customer_id').map(customer['mean']);cn=loan.index.get_level_values('customer_id').map(customer['count'])
 N=len(rp);L=len(loan);I=len(customer)
 within=float((loan['sum']-loan['sum']**2/loan['count']).sum());between=float((loan['count']*(loan['mean']-cm)**2).sum())
 sigma=within/max(N-L,1);denom=float(N-(loan['count']**2/np.asarray(cn)).sum());tau=(between-sigma*(L-I))/denom if denom>0 else 0.
 rho=max(tau,0)/(max(tau,0)+sigma)if sigma+max(tau,0)>0 else 0.
 design=loan['count']*(1+(loan['count']-1)*rho);d=design.groupby(level='customer_id').sum();eff=customer['count']**2/d
 out[str(threshold)]=dict(installments=N,loans=L,borrowers=I,within_loan_variance=sigma,loan_variance_unconstrained=tau,loan_intraclass_correlation=rho,effective_to_nominal_count_quantiles=(eff/customer['count']).quantile([0,.1,.5,.9,1]).tolist(),qualification='Unbalanced nested Gaussian ANOVA moment approximation for binary observations, controlling borrower means. Not a validated Bernoulli likelihood or proof of independent loan effects.')
b.dump(dict(status='complete',target_labels_opened=False,fit_rows=50000,fit_ids_sha256=b.ids_hash(original,ix['context']),timing=timing,thresholds=out,source_sha256=b.sha(Path(__file__)),raw_sha256=b.sha(RAW/'repayment_history.csv'),seconds=time.time()-start,classification_training_admitted=False),R/'RESULT.json');print(json.dumps(out),flush=True)
