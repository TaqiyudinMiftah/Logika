"""Nine response/adaptation/periodicity hypotheses; no default-target fitting.

Population shrinkage uses correction customers from strictly earlier decision
months, with three inner folds for training queries. DPD is allowed by explicit
user instruction; unresolved source snapshot semantics remain in metadata.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import time,json,argparse
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from sklearn.model_selection import KFold
from features import ROOT,V,RAW,div
from baseline import dump,sha,ids_hash
from experiments import combined
from extension_features import beta_prior,posterior

B=V/'behavior';CAT=[];DAYS=180

def add(f,name,value,family,formula,caveat='',fitted=False,dpd=False):
 f[name]=np.asarray(value)
 CAT.append(dict(feature_name=name,family=family,formula=formula,status='FOLD_LOCAL' if fitted else 'IMPLEMENTED',
  source='account_behavior, previous_applications, selected applications, repayment_history, ecommerce_transactions',
  cutoff_rule='Own events before selected decision; complete calendar account months; population references only correction customers with selected decision before query month',
  leakage_risk='DPD snapshot provenance unresolved; use allowed by user' if dpd else 'Incrementally low; inherited baseline uncertainty remains',
  caveat=caveat,validation_status='Unvalidated',priority='P1'))

def centered_stats(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);n=len(x)
 if n==0:return np.zeros(4)
 return np.array([n,np.sum((x-x.mean())**2),np.sum((x-x.mean())*(y-y.mean())),np.sum((y-y.mean())**2)])

def response_prior(reference,prefix):
 n,xx,xy,yy=[reference[prefix+c].to_numpy(float) for c in ['n','xx','xy','yy']]
 ok=(n>=3)&(xx>1e-10)&np.isfinite(xx+xy+yy)
 if not ok.any():return dict(mu=1.,penalty=1.,noise=1.,fallback=True)
 mu=float(xy[ok].sum()/xx[ok].sum());penalty=max(float(np.median(xx[ok])),.01)
 noise=max(float(np.sum(np.maximum(yy[ok]-2*mu*xy[ok]+mu**2*xx[ok],0))/np.maximum(n[ok]-2,1).sum()),1e-6)
 return dict(mu=mu,penalty=penalty,noise=noise,fallback=False)

def response_apply(query,prefix,prior):
 n,xx,xy=[query[prefix+c].to_numpy(float) for c in ['n','xx','xy']]
 ok=(n>=3)&(xx>1e-10);mean=(xy+prior['penalty']*prior['mu'])/(xx+prior['penalty'])
 sd=np.sqrt(prior['noise']/(xx+prior['penalty']))
 return np.where(ok,mean,np.nan),np.where(ok,sd,np.nan)

def asymmetric_moments(x,y):
 u=np.maximum(x,0);d=np.minimum(x,0);n=len(x)
 return [n,u.sum(),d.sum(),np.dot(u,u),np.dot(d,d),y.sum(),np.dot(u,y),np.dot(d,y),np.dot(y,y),(x>0).sum(),(x<0).sum()]

def asymmetric_matrices(query):
 v=query[[f'beh_aux_asym_{c}' for c in ['n','u','d','uu','dd','y','uy','dy','yy','up','down']]].to_numpy(float)
 n,u,d,uu,dd,y,uy,dy,yy,up,down=v.T;nz=np.maximum(n,1)
 gram=np.zeros((len(query),2,2));gram[:,0,0]=uu-u*u/nz;gram[:,1,1]=dd-d*d/nz
 gram[:,0,1]=gram[:,1,0]=-u*d/nz
 rhs=np.column_stack([uy-u*y/nz,dy-d*y/nz]);ss=np.maximum(yy-y*y/nz,0)
 ok=(up>=2)&(down>=2)&(n>=4)
 return gram,rhs,ss,n,ok

def asymmetric_prior(reference):
 gram,rhs,ss,n,ok=asymmetric_matrices(reference)
 if not ok.any():return dict(mu=np.ones(2),penalty=1.,noise=1.,fallback=True)
 mu=np.linalg.solve(gram[ok].sum(axis=0)+np.eye(2)*.01,rhs[ok].sum(axis=0))
 penalty=max(float(np.median(np.trace(gram[ok],axis1=1,axis2=2)/2)),.01)
 residual=ss[ok]-2*rhs[ok]@mu+np.einsum('i,nij,j->n',mu,gram[ok],mu)
 return dict(mu=mu,penalty=penalty,noise=max(float(np.maximum(residual,0).sum()/np.maximum(n[ok]-3,1).sum()),1e-6),fallback=False)

def asymmetric_apply(query,prior):
 gram,rhs,_,_,ok=asymmetric_matrices(query);cov=np.linalg.inv(gram+np.eye(2)[None]*prior['penalty'])
 beta=np.einsum('nij,nj->ni',cov,rhs+prior['penalty']*prior['mu'])
 variance=prior['noise']*(cov[:,0,0]+cov[:,1,1]-2*cov[:,0,1])
 beta[~ok]=np.nan
 return beta,np.where(ok,np.sqrt(np.maximum(variance,0)),np.nan)

def logistic_fit(x,k,n,prior,penalty,iterations=15):
 """Two-coefficient weighted-binomial MAP; each loan has unit total exposure."""
 design=np.column_stack([np.ones(len(x)),x]);beta=np.asarray(prior,float).copy();penalty=np.asarray(penalty,float)
 for _ in range(iterations):
  p=expit(design@beta);h=design.T@((n*p*(1-p))[:,None]*design)+np.diag(penalty)
  gradient=design.T@(n*p-k)+penalty*(beta-prior);step=np.linalg.solve(h,gradient)
  beta-=np.clip(step,-1,1)
  if np.max(abs(step))<1e-6:break
 return beta,np.linalg.inv(h)

def spectral(y):
 """Hann-tapered linearly detrended daily periodogram, normalized band power."""
 y=np.asarray(y,float);t=np.arange(len(y),dtype=float);t-=t.mean()
 detrended=y-y.mean()-t*np.dot(t,y)/np.dot(t,t)
 power=abs(np.fft.rfft(detrended*np.hanning(len(y))))**2;frequency=np.fft.rfftfreq(len(y));power[0]=0
 total=power.sum()
 if total<=1e-10:return np.full(5,np.nan)
 period=np.divide(1,frequency,out=np.full_like(frequency,np.inf),where=frequency>0)
 bands=[power[(period>=lo)&(period<=hi)].sum()/total for lo,hi in [(6,8),(12,16),(25,35)]]
 eligible=(period>=3)&(period<=90);peak=np.flatnonzero(eligible)[np.argmax(power[eligible])]
 return np.r_[bands,power[peak]/total,period[peak]]

def allan(y,tau):
 y=np.asarray(y,float);blocks=len(y)//tau
 if blocks<4 or y.mean()<=0:return np.nan
 means=y[:blocks*tau].reshape(blocks,tau).mean(axis=1)
 return .5*np.mean(np.diff(means)**2)/(y.mean()**2)

def match_recovery(g,cutoff):
 """Greedy chronological one-to-one compatible order matching; 7d follow-up."""
 g=g.sort_values('transaction_datetime',kind='stable');used=set();hits=[];delays=[]
 cancelled=g[(g.order_status=='cancelled')&(g.transaction_datetime<=cutoff-pd.Timedelta(days=7))&(g.transaction_datetime>=cutoff-pd.Timedelta(days=90))]
 completed=g[g.order_status=='completed']
 for c in cancelled.itertuples():
  q=completed[(completed.transaction_datetime>c.transaction_datetime)&(completed.transaction_datetime<=c.transaction_datetime+pd.Timedelta(days=7))]
  for candidate in q.itertuples():
   if candidate.Index in used or c.order_total<=0:continue
   overlap=(int(c.mask)&int(candidate.mask)).bit_count()/max((int(c.mask)|int(candidate.mask)).bit_count(),1)
   if overlap<.5 or abs(candidate.order_total/c.order_total-1)>.25:continue
   used.add(candidate.Index);hits.append(c.Index);delays.append((candidate.transaction_datetime-c.transaction_datetime).total_seconds()/86400);break
 n=len(cancelled)
 return n,len(hits),sum(d<=1 for d in delays),sum(d<=3 for d in delays),float(np.mean(delays)) if delays else np.nan

def build():
 start=time.time();(B/'data').mkdir(parents=True,exist_ok=True);(B/'reports').mkdir(exist_ok=True)
 a=pd.read_parquet(V/'discovery/data/applications.parquet');a['decision_datetime']=pd.to_datetime(a.decision_datetime);a['application_datetime']=pd.to_datetime(a.application_datetime);a.index.name='row'
 assert a.customer_id.is_unique and a.application_id.is_unique and 'target' not in a
 f=pd.DataFrame(dict(application_id=a.application_id,beh_aux_row=np.arange(len(a)),beh_aux_decision_month=a.decision_datetime.dt.to_period('M').astype(int)),index=a.index)
 lookup=a.reset_index()[['row','customer_id','decision_datetime']];align=lambda s,fill=np.nan:s.reindex(f.index,fill_value=fill)
 ac=pd.read_csv(RAW/'account_behavior.csv',engine='pyarrow');ac['month']=pd.to_datetime(ac.month)
 assert not ac.duplicated(['customer_id','month']).any();ac=ac.merge(lookup,on='customer_id',validate='many_to_one')
 ac=ac[ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime].copy();ac['period']=ac.month.dt.to_period('M').astype(int)
 ac=ac[ac.period>=ac.decision_datetime.dt.to_period('M').astype(int)-12].sort_values(['row','month'])
 mpc=np.zeros((len(a),4));lag=mpc.copy();asym=np.zeros((len(a),11));absorb=np.full(len(a),np.nan)
 for row,g in ac.groupby('row',sort=False):
  income=g.income_like_credits.to_numpy(float);out=g.outflows.to_numpy(float);period=g.period.to_numpy();scale=np.median(income[income>0]) if (income>0).any() else np.nan
  if not np.isfinite(scale):continue
  x=np.diff(income)/scale;y=np.diff(out)/scale;ok=np.diff(period)==1
  mpc[row]=centered_stats(x[ok],y[ok]);asym[row]=asymmetric_moments(x[ok],y[ok])
  lagok=ok[:-1]&ok[1:];lag[row]=centered_stats(x[:-1][lagok],y[1:][lagok])
  down=ok&(x<0)
  if down.sum()>=2:absorb[row]=float(-y[down].sum()/(-x[down]).sum())
 for mat,prefix in [(mpc,'mpc'),(lag,'lag')]:
  for j,c in enumerate(['n','xx','xy','yy']):f[f'beh_aux_{prefix}_{c}']=mat[:,j]
 for j,c in enumerate(['n','u','d','uu','dd','y','uy','dy','yy','up','down']):f['beh_aux_asym_'+c]=asym[:,j]
 for name,formula in [('outflow_income_sensitivity','(within-customer centered sum(dx*dy)+lambda*beta_population)/(sum(dx^2)+lambda)'),('spending_response_lag1','same posterior slope for dx_t and dy_(t+1), contiguous triples only'),('savings_capture_rate','1 - posterior contemporaneous response slope'),('sensitivity_uncertainty','sqrt(population residual variance/(centered sum(dx^2)+lambda))')]:
  add(f,'beh_mpc_'+name,np.full(len(a),np.nan),'mpc',formula+'; dx=Delta income_like_credits/own median positive income, dy=Delta outflows/same scale; >=3 valid changes',
   'Associational response, not causal MPC; complete observed months only, no interpolation. Reference priors exclude later decision months.',True)
 add(f,'beh_mpc_observed_changes',mpc[:,0],'mpc','number adjacent complete-month changes used; maximum12 observed months')
 for name,formula in [('upside_elasticity','MAP beta_up in dy=intercept+beta_up*max(dx,0)+beta_down*min(dx,0)'),('downside_elasticity','MAP beta_down in same response model'),('response_asymmetry','beta_up-beta_down'),('asymmetry_uncertainty','sqrt(sigma2*(inverse_precision_uu+inverse_precision_dd-2*inverse_precision_ud))')]:
  add(f,'beh_asym_'+name,np.full(len(a),np.nan),'asymmetric',formula+'; ridge toward dated correction-population coefficients; >=2 positive and2 negative income changes',
   'Scaled-change response coefficient, not log-log causal elasticity; intercept is analytically centered/unpenalized.',True)
 add(f,'beh_asym_shock_absorption_ratio',absorb,'asymmetric','sum(-Delta_outflow)/sum(-Delta_income) for negative income changes; >=2 declines; no clipping')
 add(f,'beh_asym_income_up_count',asym[:,9],'asymmetric','number positive income changes')
 add(f,'beh_asym_income_down_count',asym[:,10],'asymmetric','number negative income changes')
 print('Behavior account-response sufficient statistics READY',flush=True)
 prev=pd.read_csv(RAW/'previous_applications.csv',engine='pyarrow');prev['historical_decision']=pd.to_datetime(prev.pop('decision_datetime'));prev['application_datetime']=pd.to_datetime(prev.application_datetime)
 prev=prev.merge(lookup,on='customer_id',validate='many_to_one');prev=prev[(prev.historical_decision<prev.decision_datetime)&~prev.application_id.isin(a.application_id)].copy()
 assert (prev.application_datetime<=prev.historical_decision).all();prev=prev.sort_values(['row','application_datetime','historical_decision'])
 prev['next_application']=prev.groupby('row').application_datetime.shift(-1);prev['next_amount']=prev.groupby('row').requested_amount.shift(-1);prev['next_decision']=prev.groupby('row').decision.shift(-1)
 reject=prev[prev.decision=='rejected'].copy();paired=reject[(reject.next_application>reject.historical_decision)&(reject.next_amount>0)&(reject.requested_amount>0)].copy()
 paired['concession']=1-paired.next_amount/paired.requested_amount;paired['wait']=(paired.next_application-paired.historical_decision).dt.total_seconds()/86400
 paired['escalate']=paired.next_amount>paired.requested_amount;paired['success_concession']=(paired.concession>0)&(paired.next_decision=='approved')
 for name,value,formula in [('amount_concession',paired.groupby('row').concession.mean(),'mean(1-next_requested/rejected_requested); negative means escalation'),('waiting_days',paired.groupby('row').wait.median(),'median days between rejection decision and next historical application'),('escalation_index',paired.groupby('row').escalate.mean(),'share eligible post-rejection pairs with increased requested amount')]:
  add(f,'beh_reapply_'+name,align(value),'reapplication',formula+'; adjacent historical applications only; prior rejection must precede next submission')
 eligible=reject[reject.historical_decision<=reject.decision_datetime-pd.Timedelta(days=30)].copy()
 eligible['within30']=(eligible.next_application>eligible.historical_decision)&(eligible.next_application<=eligible.historical_decision+pd.Timedelta(days=30))
 add(f,'beh_reapply_after_rejection_intensity',align(eligible.groupby('row').within30.mean()),'reapplication','fraction rejected historical decisions followed by next application within30d; full30d follow-up before cutoff',
  'Observation platform coverage is not documented; measures recorded reapplication, not total external credit search.')
 concede=paired[paired.concession>0]
 f['beh_aux_reapply_k']=align(concede.groupby('row').success_concession.sum(),0);f['beh_aux_reapply_n']=align(concede.groupby('row').size(),0)
 add(f,'beh_reapply_success_after_concession',np.full(len(a),np.nan),'reapplication','Beta-Binomial posterior approval rate for observed next historical applications that reduced requested amount after rejection; n>0',fitted=True)
 add(f,'beh_reapply_pair_count',align(paired.groupby('row').size(),0),'reapplication','number observed eligible adjacent post-rejection pairs')
 loan=prev[(prev.decision=='approved')&prev.loan_id.notna()].copy();assert loan.loan_id.is_unique
 loan['rho']=div(loan.approved_amount,loan.requested_amount);ration=np.full((len(a),4),np.nan)
 for row,g in loan.groupby('row',sort=False):
  g=g.sort_values('historical_decision');x=(g.historical_decision-g.historical_decision.iloc[0]).dt.total_seconds().to_numpy()/86400/180;y=g.rho.to_numpy(float);valid=np.isfinite(y);x,y=x[valid],y[valid]
  if len(y)>=2:
   ration[row,0]=y[-1]-y[0]
   if np.ptp(x)>=1/6:ration[row,1]=centered_stats(x,y)[2]/max(centered_stats(x,y)[1],1e-10)
  if len(y):ration[row,2]=y[-1];ration[row,3]=len(y)
 for j,(name,formula) in enumerate([('first_last_change','last approved/requested ratio minus first ratio'),('trajectory_per180d','OLS slope(rho,time in180-day units); >=2 approvals, span>=30d'),('latest_ratio','last historical approved_amount/requested_amount'),('approved_observations','number historical approved observations with defined ratio')]):
  add(f,'beh_rationing_'+name,ration[:,j],'rationing',formula,'Approved-only selection and changing requested amounts confound institutional credit-supply interpretation; no causal rationing claim.')
 prev['delay']=(prev.historical_decision-prev.application_datetime).dt.total_seconds()/60
 delay=prev.sort_values(['row','historical_decision']).groupby('row').delay.agg(['median','first','last','size']);by=prev.groupby(['row','decision']).delay.median().unstack('decision').reindex(columns=['approved','rejected'])
 current=(a.decision_datetime-a.application_datetime).dt.total_seconds()/60;assert (current>=0).all()
 for name,value,formula in [('historical_median',align(delay['median']),'median historical decision lag minutes'),('historical_change',align(delay['last']-delay['first']),'latest-minus-first historical decision lag'),('approval_rejection_gap',align(by.approved-by.rejected),'median approval delay minus median rejection delay'),('current_vs_history',current-align(delay['median']),'selected decision lag minus own historical median; known at decision cutoff')]:
  add(f,'beh_friction_'+name,value,'friction',formula,'Workflow/policy artifact possible; compare duration distribution, cohort stability and constant features before interpreting risk.')
 print('Behavior credit adaptation, rationing and friction READY',flush=True)
 orders=pd.read_parquet(V/'discovery/data/orders.parquet');orders['transaction_datetime']=pd.to_datetime(orders.transaction_datetime).astype('datetime64[ns]')
 assert (orders.age>0).all();first=orders.groupby('row').transaction_datetime.min();end=a.decision_datetime.dt.normalize();begin=end-pd.Timedelta(days=DAYS)
 support=align(first)<=begin;orders['bin']=(orders.transaction_datetime.dt.normalize()-orders.row.map(begin)).dt.days
 q=orders[(orders.order_status=='completed')&orders.bin.between(0,DAYS-1)].copy()
 spend=np.zeros((len(a),DAYS),dtype=np.float32);counts=np.zeros((len(a),DAYS),dtype=np.float32)
 ix=(q.row.to_numpy(int),q.bin.to_numpy(int));np.add.at(spend,ix,q.order_total.to_numpy(float));np.add.at(counts,ix,1.)
 sf=np.full((len(a),10),np.nan);av=np.full((len(a),5),np.nan)
 enough=support.to_numpy()&(counts.sum(axis=1)>=8)
 for row in np.flatnonzero(enough):
  sf[row,:5]=spectral(spend[row]);sf[row,5:]=spectral(counts[row]);v=[allan(spend[row],tau) for tau in [1,7,14]]
  av[row,:3]=v
  if v[0]>0:av[row,3]=v[1]/v[0];av[row,4]=v[2]/v[0]
 for offset,channel in [(0,'spend'),(5,'orders')]:
  for j,name in enumerate(['power_7d','power_14d','power_30d','periodicity_strength','dominant_cycle_days']):
   add(f,'beh_spectral_'+channel+'_'+name,sf[:,offset+j],'spectral','Hann-window detrended180-full-day '+channel+' periodogram; '+name+'; bands6:8d,12:16d,25:35d; dominant period3:90d',
    'At least8 completed orders and first recorded event before180d window; first event is only coverage proxy. Current incomplete calendar day excluded.')
 for j,name in enumerate(['variance_1d','variance_7d','variance_14d','weekly_daily_ratio','fortnight_daily_ratio']):
  add(f,'beh_allan_'+name,av[:,j],'allan','Avar(tau)=.5*mean((adjacent nonoverlapping tau-day block means difference)^2)/mean(daily_spend)^2; '+name+'; >=4 blocks',
   'Fixed180-full-day window, no overlapping blocks; scale-normalized, finite-history estimator, not a demonstrated financial mechanism.')
 add(f,'beh_spectral_window_supported',support.astype(float),'spectral','1 if first recorded transaction predates180-day full-calendar-day window')
 add(f,'beh_allan_order_support',counts.sum(axis=1),'allan','number completed orders in same180-full-day window')
 del spend,counts
 recover=np.zeros((len(a),5));recover[:,4]=np.nan
 for row,g in orders.groupby('row',sort=False):recover[row]=match_recovery(g,a.decision_datetime.iloc[row])
 for j,days in [(1,7),(2,1),(3,3)]:
  add(f,f'beh_cancel_recovery_{days}d',div(recover[:,j],recover[:,0]),'cancel_recovery',f'matched completed orders within{days}d / eligible cancelled orders in last90d with full7d follow-up',
   'Greedy one-to-one chronological matching; category-mask Jaccard>=.5 and amount within25%; reasons for cancellation unknown.')
 add(f,'beh_cancel_recovery_delay',recover[:,4],'cancel_recovery','mean days to compatible completed order within7d; missing without matched event')
 add(f,'beh_cancel_recovery_exposure',recover[:,0],'cancel_recovery','eligible cancelled-order count with full7-day follow-up')
 print('Behavior spectral, Allan and cancelled-order recovery READY',flush=True)
 # Loan-normalized binomial evidence for individual size-response curves.
 r=pd.read_csv(RAW/'repayment_history.csv',engine='pyarrow');original=len(r);assert r.installment_id.is_unique
 r['due_date']=pd.to_datetime(r.due_date);r=r.merge(loan[['loan_id','customer_id','row','decision_datetime']],on=['loan_id','customer_id'],validate='many_to_one')
 assert len(r)==original and r.row.notna().all();r['age']=(r.decision_datetime.dt.normalize()-r.due_date.dt.normalize()).dt.days
 r=r[(r.age>=15)&(r.amount_due>0)].copy();r['bad']=r.max_days_past_due>=15
 center=r.groupby('row').amount_due.median();r['x']=np.log(r.amount_due/r.row.map(center))
 bins=r.groupby(['row','loan_id','x'],as_index=False).agg(n=('bad','size'),k=('bad','sum'))
 loan_n=bins.groupby(['row','loan_id']).n.transform('sum');bins['n']=bins.n/loan_n;bins['k']=bins.k/loan_n
 bins.to_parquet(B/'data/size_bins.parquet',index=False);f['beh_aux_size_center']=align(center)
 f['beh_aux_current_installment']=a.proposed_monthly_installment
 for col,formula in [('delinquency_slope','individual MAP beta in logit P(maxDPD>=15)=intercept+beta*log(amount_due/own_median_due)'),('high_installment_failure_odds','beta*(P90_logsize-P10_logsize), log odds contrast'),('size_sensitivity_confidence','posterior slope SD from inverse penalized Hessian; smaller means stronger evidence'),('current_installment_extrapolation','sigmoid(intercept+beta*log(new_installment/own_median_due))')]:
  add(f,'beh_repay_'+col,np.full(len(a),np.nan),'size_elasticity',formula+'; mature15d installments; per-loan total binomial weight1; >=2 loans and distinct sizes',
   'DPD allowed by explicit user instruction; snapshot provenance unresolved. Current installment may be outside observed range; not a calibrated future-default PD.',True,True)
 add(f,'beh_repay_size_observed_loans',align(bins.groupby('row').loan_id.nunique(),0),'size_elasticity','number loans contributing mature15d installment evidence',dpd=True)
 for c in f.columns.difference(['application_id']):f[c]=pd.to_numeric(f[c]).replace([np.inf,-np.inf],np.nan)
 f.to_parquet(B/'data/raw_features.parquet',index=False);pd.DataFrame(CAT).to_csv(B/'reports/FEATURE_CATALOG.csv',index=False)
 dump(dict(status='READY',rows=len(f),catalog_features=len(CAT),feature_sha256=sha(B/'data/raw_features.parquet'),size_bins_sha256=sha(B/'data/size_bins.parquet'),
  elapsed_seconds=time.time()-start,no_default_labels=True,complete_account_months=True,full_followup_cancel_events=True,
  spectral_supported_customers=int(enough.sum()),DPD_allowed_by_user=True,DPD_snapshot_provenance_verified=False,
  friction_summary=dict(current_min=float(current.min()),current_max=float(current.max()),current_unique=int(current.nunique()),historical_unique=int(prev.delay.nunique()))),B/'reports/FEATURE_STATUS.json')
 print('Nine-method raw feature batch READY',flush=True)

def populate(query,reference,bins,ledger,groupbins):
 for month,positions in query.groupby('beh_aux_decision_month').groups.items():
  ids=list(positions);q=query.loc[ids];ref=reference[reference.beh_aux_decision_month<month]
  priors={key:response_prior(ref,'beh_aux_'+key+'_') for key in ['mpc','lag']};asym=asymmetric_prior(ref)
  rejection=beta_prior(ref.beh_aux_reapply_k,ref.beh_aux_reapply_n)
  for key,target in [('mpc','outflow_income_sensitivity'),('lag','spending_response_lag1')]:
   mean,sd=response_apply(q,'beh_aux_'+key+'_',priors[key]);query.loc[ids,'beh_mpc_'+target]=mean
   if key=='mpc':query.loc[ids,'beh_mpc_sensitivity_uncertainty']=sd;query.loc[ids,'beh_mpc_savings_capture_rate']=1-mean
  beta,sd=asymmetric_apply(q,asym)
  for col,value in [('upside_elasticity',beta[:,0]),('downside_elasticity',beta[:,1]),('response_asymmetry',beta[:,0]-beta[:,1]),('asymmetry_uncertainty',sd)]:query.loc[ids,'beh_asym_'+col]=value
  post,_=posterior(q.beh_aux_reapply_k,q.beh_aux_reapply_n,rejection)
  query.loc[ids,'beh_reapply_success_after_concession']=np.where(q.beh_aux_reapply_n>0,post,np.nan)
  population=bins[bins.row.isin(ref.beh_aux_row)]
  prior=np.array([logit(.15),0.]);penalty=np.array([2.,4.])
  if len(population):
   pooled=np.clip(population.k.sum()/population.n.sum(),1e-4,1-1e-4)
   prior,_=logistic_fit(population.x.to_numpy(),population.k.to_numpy(),population.n.to_numpy(),np.array([logit(pooled),0.]),np.array([10.,10.]))
  ledger.append(dict(month=int(month),reference_customers=len(ref),mpc=priors,asymmetric=asym,reapply=rejection,size_prior=prior,size_penalty=penalty))
  for pos,row in zip(ids,q.beh_aux_row.to_numpy(int)):
   if row not in groupbins:continue
   g=groupbins[row];beta,cov=logistic_fit(g.x.to_numpy(),g.k.to_numpy(),g.n.to_numpy(),prior,penalty)
   contrast=np.quantile(g.x,.9)-np.quantile(g.x,.1);new=float(div(query.loc[pos,'beh_aux_current_installment'],query.loc[pos,'beh_aux_size_center']))
   query.loc[pos,'beh_repay_delinquency_slope']=beta[1];query.loc[pos,'beh_repay_high_installment_failure_odds']=beta[1]*contrast
   query.loc[pos,'beh_repay_size_sensitivity_confidence']=np.sqrt(cov[1,1])
   if np.isfinite(new) and new>0:query.loc[pos,'beh_repay_current_installment_extrapolation']=expit(beta[0]+beta[1]*np.log(new))
 return query

def fold(number):
 folder=B/'data'/f'fold{number}';folder.mkdir(parents=True,exist_ok=True)
 raw=pd.read_parquet(B/'data/raw_features.parquet');tr=pd.read_parquet(ROOT/'data/train_features.parquet');te=pd.read_parquet(ROOT/'data/test_features.parquet')
 idx=np.load(ROOT/'artifacts'/f'fold{number}'/'indices.npz');hi,vi=idx['correction'],idx['validation'];assert not np.intersect1d(hi,vi).size
 parts={}
 for name,base in [('correction',tr.iloc[hi]),('validation',tr.iloc[vi]),('test',te)]:
  old=pd.read_parquet(V/'extension/data'/f'fold{number}'/(name+'.parquet'))
  parts[name]=combined(base,old).merge(raw,on='application_id',how='left',validate='one_to_one')
 bins=pd.read_parquet(B/'data/size_bins.parquet');h=parts['correction'];ledger=[]
 eligible=bins.groupby('row').agg(loans=('loan_id','nunique'),sizes=('x','nunique'));eligible=eligible[(eligible.loans>=2)&(eligible.sizes>=2)].index
 groupbins={int(row):g for row,g in bins[bins.row.isin(eligible)].groupby('row',sort=False)}
 for ti,qi in KFold(3,shuffle=True,random_state=31027+number).split(h):
  assert set(h.iloc[ti].beh_aux_row).isdisjoint(h.iloc[qi].beh_aux_row)
  q=populate(h.iloc[qi].copy(),h.iloc[ti],bins,ledger,groupbins);cols=[c['feature_name'] for c in pd.read_csv(B/'reports/FEATURE_CATALOG.csv').to_dict('records') if c['status']=='FOLD_LOCAL']
  h.loc[q.index,cols]=q[cols]
 for name in ['validation','test']:parts[name]=populate(parts[name],h,bins,ledger,groupbins)
 for name,d in parts.items():
  keep=['application_id']+[c for c in d if c.startswith(('v7_','v7q_','aux_','ext_aux_','risk_','riskq_','beh_'))];d[keep].to_parquet(folder/(name+'.parquet'),index=False)
 joblib.dump(dict(references=ledger,training_ids=h.application_id.tolist()),folder/'references.joblib',compress=3)
 dump(dict(status='PASS',fold=number,correction_ids_sha256=ids_hash(tr,hi),validation_ids_sha256=ids_hash(tr,vi),default_target_not_used=True,
  references_strictly_earlier_decision_month=True,inner_training_queries_disjoint=True,DPD_allowed_by_user=True,DPD_snapshot_provenance_verified=False,
  raw_sha256=sha(B/'data/raw_features.parquet')),folder/'AUDIT.json');print('Behavior fold READY',number,flush=True)

def checks():
 x=np.arange(6,dtype=float);y=2*x;stats=centered_stats(x,y);assert np.isclose(stats[2]/stats[1],2)
 n=180;wave=10+np.sin(2*np.pi*np.arange(n)/7);s=spectral(wave);assert 6<=s[-1]<=8 and s[0]>.8
 assert np.isclose(allan(np.ones(180),7),0)
 g=pd.DataFrame({'transaction_datetime':pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03']),
  'order_status':['cancelled','cancelled','completed'],'mask':[1,1,1],'order_total':[100,100,100]})
 matched=match_recovery(g,pd.Timestamp('2025-02-01'));assert matched[0]==2 and matched[1]==1
 assert match_recovery(g,pd.Timestamp('2025-01-05'))[0]==0
 b,c=logistic_fit(np.array([-1,1.]),np.array([0.,1]),np.ones(2),np.array([0.,0.]),np.array([2.,4.]));assert b[1]>0 and np.all(np.linalg.eigvalsh(c)>0)
 (B/'reports').mkdir(parents=True,exist_ok=True)
 dump(dict(status='PASS',source_sha256=sha(__file__),checks=['centered_response_slope','weekly_spectral_peak','constant_Allan_zero','recovery_one_to_one','recovery_full_followup','regularized_binomial_size_response']),B/'reports/UNIT_CHECKS.json')
 print('Nine-method integrity checks PASS',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int);p.add_argument('--checks',action='store_true');args=p.parse_args()
 if args.checks:checks()
 elif args.fold is not None:fold(args.fold)
 else:build()
