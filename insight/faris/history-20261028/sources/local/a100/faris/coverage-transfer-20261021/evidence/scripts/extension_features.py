"""Conditional liquidity, recovery and policy residuals; no default labels used."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
from pathlib import Path
import sys,json,time,argparse
import numpy as np
import pandas as pd
import joblib
from scipy.special import betaln
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from features import ROOT,V,RAW,div
from baseline import dump,sha,ids_hash
from experiments import combined,train_nuisance,nuisance_predict

E=V/'extension';CATALOG=[]

def add(frame,name,values,family,formula,caveat='',status='IMPLEMENTED'):
 frame[name]=np.asarray(values)
 CATALOG.append(dict(feature_name=name,family=family,formula=formula,status=status,
  source='Audited selected application, customer, previous applications, account, repayment and ecommerce events',
  cutoff_rule='Selected decision cutoff; complete observed account months; historical decisions only; fold-local references',
  leakage_risk='Unverified DPD snapshot' if name.startswith('riskq_') else 'Incrementally low; locked baseline retains inherited uncertainties',
  caveat=caveat,validation_status='Unvalidated',priority='P0' if family in ['conditional_allocation','cliff','recovery'] else 'P1'))

def recovery_statistics(period,net,opening,closing):
 """Only truly consecutive observed future months may resolve an episode."""
 one=[];three=[];times=[];balance_fail=[];total=0
 for j in range(len(net)):
  if net[j]>=0:continue
  total+=1
  if j+1<len(net) and period[j+1]-period[j]==1:
   one.append(float(net[j+1]>=0 and closing[j+1]>=opening[j]))
  if j+3<len(net) and np.all(np.diff(period[j:j+4])==1):
   nextnet=net[j+1:j+4];positive=np.flatnonzero(nextnet>=0)
   times.append(float(positive[0]+1) if len(positive) else 3.)
   three.append(float(not np.any(nextnet>=0)))
   balance_fail.append(float(closing[j+3]<opening[j]))
 return dict(probability=np.mean(one) if one else np.nan,time=np.mean(times) if times else np.nan,
  failure=np.mean(three) if three else np.nan,balance_failure=np.mean(balance_fail) if balance_fail else np.nan,
  episodes=total,one_observed=len(one),three_observed=len(three))

def beta_prior(k,n):
 k,n=np.asarray(k,float),np.asarray(n,float);ok=(n>0)&np.isfinite(k)&np.isfinite(n);k,n=k[ok],n[ok]
 assert np.all((k>=0)&(k<=n))
 if len(n)==0:return dict(alpha=.5,beta=.5,source='No exposed training customers; noninformative fallback',customers=0)
 mean=np.clip((k.sum()+.5)/(n.sum()+1),1e-4,1-1e-4)
 if len(n)<50:return dict(alpha=float(mean*10),beta=float((1-mean)*10),source='Prespecified strength10, training pooled mean; sparse population',customers=len(n))
 def objective(z):
  a,b=np.exp(z);return -float(np.sum(betaln(a+k,b+n-k)-betaln(a,b)))
 fit=minimize(objective,np.log([mean*10,(1-mean)*10]),method='L-BFGS-B',bounds=[(-5,9),(-5,9)])
 a,b=np.exp(fit.x)
 if not fit.success:a,b=mean*10,(1-mean)*10
 return dict(alpha=float(a),beta=float(b),source='Training-only Beta-Binomial MLE' if fit.success else 'Training pooled fallback',customers=len(n),optimizer_success=bool(fit.success))

def posterior(k,n,prior):
 k,n=np.asarray(k,float),np.asarray(n,float);a=prior['alpha']+k;b=prior['beta']+n-k;den=a+b
 return a/den,a*b/(den*den*(den+1))

def build():
 started=time.time();(E/'data').mkdir(parents=True,exist_ok=True);(E/'reports').mkdir(exist_ok=True)
 a=pd.concat([pd.read_csv(RAW/'train_applications.csv').drop(columns='target'),pd.read_csv(RAW/'test_applications.csv')],ignore_index=True)
 a['decision_datetime']=pd.to_datetime(a.decision_datetime);assert a.application_id.is_unique and a.customer_id.is_unique;a.index.name='row'
 lookup=a.reset_index()[['row','customer_id','decision_datetime']];f=pd.DataFrame({'application_id':a.application_id},index=a.index)
 ac=pd.read_csv(RAW/'account_behavior.csv',engine='pyarrow');ac['month']=pd.to_datetime(ac.month)
 assert not ac.duplicated(['customer_id','month']).any()
 ac=ac.merge(lookup,on='customer_id',validate='many_to_one');ac=ac[ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime].copy()
 ac['net']=ac.inflows-ac.outflows;ac=ac.sort_values(['row','month']);ac['period']=ac.month.dt.to_period('M').astype(int)
 start=ac.decision_datetime.dt.to_period('M').dt.to_timestamp()-pd.DateOffset(months=3)
 recent=ac[ac.month>=start].groupby('row').agg(income=('income_like_credits','mean'),net=('net','mean'),outflow=('outflows','mean'))
 income=recent.income.reindex(a.index);net=recent.net.reindex(a.index)
 new_i=a.proposed_monthly_installment
 cliff=div(new_i,net)
 add(f,'risk_debt_payment_cliff_net_new',np.maximum(cliff-1,0)**2,'cliff','max(new_installment/positive_observed_net_cashflow_3m-1,0)^2; nonpositive CF -> missing',
  'Observed net flow may already include old debt; only new installment is added. Cash visibility is partial.')
 add(f,'risk_debt_payment_cliff_nonpositive_capacity',np.where(net.notna(),(net<=0).astype(float),np.nan),'cliff','1 if observed mean net cashflow_3m<=0; NA with no observations')
 add(f,'risk_debt_payment_cliff_shortfall',div(np.maximum(new_i-net,0),income),'cliff','max(new_installment-observed_net_CF_3m,0)/observed_income_3m')
 # Contractual active schedule is known at origination; early settlement is not inferable.
 prev=pd.read_csv(RAW/'previous_applications.csv',engine='pyarrow');prev['historical_decision']=pd.to_datetime(prev.pop('decision_datetime'))
 prev=prev.merge(lookup,on='customer_id',validate='many_to_one');prev=prev[(prev.historical_decision<prev.decision_datetime)&(~prev.application_id.isin(a.application_id))].copy()
 originated=prev[(prev.decision=='approved')&prev.loan_id.notna()].copy()
 number=originated.historical_decision.dt.year*12+originated.historical_decision.dt.month-1+originated.term_months
 first=pd.to_datetime(pd.DataFrame({'year':number//12,'month':number%12+1,'day':1},index=originated.index))
 maturity=first+pd.to_timedelta(np.minimum(originated.historical_decision.dt.day,first.dt.days_in_month)-1,unit='D')+(originated.historical_decision-originated.historical_decision.dt.normalize())
 active=originated[maturity>originated.decision_datetime]
 scheduled=active.groupby('row').scheduled_monthly_installment.sum().reindex(a.index,fill_value=0)
 f['ext_aux_contractual_active_scheduled']=scheduled
 add(f,'risk_debt_payment_cliff_gross_scheduled',np.maximum(div(new_i+scheduled,income)-1,0)**2,'cliff','max((new_installment+contractually_active_scheduled_installments)/observed_income_3m-1,0)^2',
  'Gross income denominator avoids adding existing installments to net CF. Contractual active is not verified outstanding; prepayments unknown.')
 f['ext_aux_contractual_active_loans']=active.groupby('row').size().reindex(a.index,fill_value=0)
 # Account recovery episodes, with explicit censoring and consecutive-month checks.
 recovery=pd.DataFrame(index=a.index,columns=['probability','time','failure','balance_failure','episodes','one_observed','three_observed'],dtype=float)
 for count,(i,g) in enumerate(ac.groupby('row',sort=False)):
  recovery.loc[i]=recovery_statistics(g.period.to_numpy(),g.net.to_numpy(),g.opening_balance.to_numpy(),g.closing_balance.to_numpy())
  if count%25000==0:print('Extension recovery customers',count,flush=True)
 for col,name,formula in [
  ('probability','risk_recovery_probability_after_deficit','mean(next_month_net>=0 AND next_month_closing>=deficit_month_opening), among deficit months with observed next calendar month'),
  ('time','risk_recovery_time_to_positive_cashflow','mean(min(first positive-flow future month,3)), among deficit episodes with three consecutive future months; uncured capped at3'),
  ('failure','risk_recovery_failure_3m','share of fully followed deficit episodes with no nonnegative-flow month in following three months'),
  ('balance_failure','risk_liquidity_recovery_failure','share of fully followed deficit episodes whose month+3 closing balance remains below deficit-month opening balance')]:
  add(f,name,recovery[col],'recovery',formula,'Follow-up must be observed and completed before cutoff; absence of follow-up is missing, not failure. Episodes may overlap, so trials are not independent.')
 for col in ['episodes','one_observed','three_observed']:add(f,'risk_recovery_'+col,recovery[col],'recovery','Number of '+col+' deficit episodes; missing account history stays missing')
 # Spending uses item_total, only completed item events, in the exact account months.
 ec=pd.read_csv(RAW/'ecommerce_transactions.csv',usecols=['customer_id','transaction_datetime','item_total','order_status'],engine='pyarrow')
 ec['transaction_datetime']=pd.to_datetime(ec.transaction_datetime);ec=ec.merge(lookup,on='customer_id',validate='many_to_one')
 ec=ec[(ec.transaction_datetime<ec.decision_datetime)&(ec.order_status=='completed')].copy();ec['month']=ec.transaction_datetime.dt.to_period('M').dt.to_timestamp()
 em=ec.groupby(['row','month']).item_total.sum().rename('spend').reset_index();oldest=ec.groupby('row').transaction_datetime.min()
 monthly=ac.merge(em,on=['row','month'],how='left',validate='one_to_one');monthly['spend']=monthly.spend.fillna(0)
 decouple=np.full((len(a),3),np.nan)
 for i,g in monthly.groupby('row',sort=False):
  start=a.loc[i,'decision_datetime'].to_period('M').to_timestamp()-pd.DateOffset(months=6)
  g=g[g.month>=start]
  if len(g)!=6 or not np.all(np.diff(g.period)==1) or i not in oldest or oldest.loc[i]>start:continue
  old=g.iloc[:3];new=g.iloc[3:]
  inc_growth=np.log(div(new.income_like_credits.mean(),old.income_like_credits.mean()))
  spend_growth=np.log(div(new.spend.mean(),old.spend.mean()))
  if np.isfinite(inc_growth) and np.isfinite(spend_growth):decouple[i]=[max(spend_growth,0)*max(-inc_growth,0),spend_growth-inc_growth,float(spend_growth>0 and inc_growth<0)]
 for j,name,formula in [(0,'risk_income_spending_decoupling','max(log(spend_recent3/spend_prior3),0)*max(-log(income_recent3/income_prior3),0)'),
  (1,'risk_income_spending_growth_gap','log(spend_recent3/spend_prior3)-log(income_recent3/income_prior3)'),
  (2,'risk_income_down_spending_up','1 if completed-month spend grows while observed income falls')]:
  add(f,name,decouple[:,j],'decoupling',formula,'Requires six consecutive account months and recorded ecommerce history before window start; zero denominator undefined. Provider enrollment itself is unknown.')
 # Monthly conditional events: one observation per borrower-month, not per installment.
 r=pd.read_csv(RAW/'repayment_history.csv',engine='pyarrow');r['due_date']=pd.to_datetime(r.due_date);r['last_payment_datetime']=pd.to_datetime(r.last_payment_datetime)
 r=r.merge(lookup,on='customer_id',validate='many_to_one');r=r[(r.due_date<r.decision_datetime)].copy()
 assert (r.last_payment_datetime.isna()|(r.last_payment_datetime<r.decision_datetime)).all()
 r['age']=(r.decision_datetime.dt.normalize()-r.due_date.dt.normalize()).dt.days;r['month']=r.due_date.dt.to_period('M').dt.to_timestamp()
 r['allocation_delay']=(r.last_payment_datetime-r.due_date).dt.total_seconds()/86400
 rows=r[r.age>=15].copy();cf=ac[['row','month','net']]
 for kind in ['allocation','dpd']:
  eligible=rows[rows.last_payment_datetime.notna()].copy() if kind=='allocation' else rows.copy()
  eligible['event']=(eligible.allocation_delay>=15) if kind=='allocation' else (eligible.max_days_past_due>=15)
  observed=eligible.groupby(['row','month']).event.max().reset_index().merge(cf,on=['row','month'],how='inner',validate='one_to_one')
  for condition,mask in [('negative',observed.net<0),('nonnegative',observed.net>=0)]:
   counts=observed[mask].groupby('row').event.agg(['sum','size']).reindex(a.index,fill_value=0)
   f[f'ext_aux_{kind}_{condition}_k']=counts['sum'];f[f'ext_aux_{kind}_{condition}_n']=counts['size']
 # Only timestamps can order deterioration before a historical rejection.
 loan_events={i:g.sort_values('last_payment_datetime') for i,g in r[r.last_payment_datetime.notna()&(r.allocation_delay>=15)].groupby('row',sort=False)}
 access=np.full((len(a),3),np.nan)
 for i,g in prev.groupby('row',sort=False):
  g=g.sort_values('historical_decision');approved=g[g.decision=='approved'].historical_decision.to_numpy();rejected=g[g.decision=='rejected'].historical_decision.to_numpy()
  if not len(g):continue
  transitions=0;ordered_late=0
  for rejection in rejected:
   earlier=approved[approved<rejection]
   if not len(earlier):continue
   transitions+=1
   if i in loan_events:
    dates=loan_events[i].last_payment_datetime.to_numpy()
    ordered_late+=int(np.any((dates>earlier[-1])&(dates<rejection)))
  access[i]=[float(transitions),float(ordered_late),div(ordered_late,len(rejected))]
 for j,name,formula in [(0,'risk_credit_access_approval_to_rejection_count','count historical rejections preceded by historical approval'),
  (1,'risk_credit_access_deterioration_allocation_proxy','count approval -> observed >=15d delayed last allocation -> rejection timestamp sequences'),
  (2,'risk_credit_access_late_sequence_share','ordered late-allocation sequences / historical rejection count')]:
  add(f,name,access[:,j],'credit_sequence',formula,'Last-allocation events are not complete payment trajectories or proof DPD worsened before rejection. No historical current-snapshot DPD is backdated.')
 # Inputs for fold-local underwriting and disagreement nuisance regressions.
 f['ext_aux_interest_rate']=a.annual_interest_rate
 f['ext_aux_dpd_mean']=r.groupby('row').max_days_past_due.mean().reindex(a.index)
 recent_r=r[(r.age>=15)&(r.age<=90)]
 f['ext_aux_recent_dpd15_share']=recent_r.assign(late=recent_r.max_days_past_due>=15).groupby('row').late.mean().reindex(a.index)
 f['ext_aux_recent_dpd15_n']=recent_r.groupby('row').size().reindex(a.index,fill_value=0)
 for prefix,family,status in [('risk_','conditional_allocation','FOLD_LOCAL'),('riskq_','conditional_dpd','QUARANTINED_FOLD_LOCAL')]:
  for suffix,formula in [('conditional_late_negative','Beta-Binomial posterior mean for event months with negative due-month CF'),('conditional_late_nonnegative','Beta-Binomial posterior mean for event months with nonnegative due-month CF'),
   ('dpd_cashflow_dependency' if prefix=='riskq_' else 'allocation_cashflow_dependency','posterior_mean_negative - posterior_mean_nonnegative; NA unless both exposures observed'),
   ('conditional_dependency_sd','sqrt(posterior_variance_negative+posterior_variance_nonnegative), assuming independent conditional propensities'),
   ('conditional_negative_months','number of eligible observed negative-CF months'),('conditional_nonnegative_months','number of eligible observed nonnegative-CF months')]:
   CATALOG.append(dict(feature_name=prefix+suffix,family=family,formula=formula,status=status,source='Monthly eligible repayment events aligned to complete due-month account cashflow',cutoff_rule='Events observed at selected cutoff; priors fitted on correction-training borrowers only',leakage_risk='Unverified snapshot' if prefix=='riskq_' else 'Incrementally low',caveat='Allocation delay is not DPD. DPD is maximum by selected cutoff, not necessarily observed during due month. Monthly dependence can understate posterior uncertainty.',validation_status='Unvalidated',priority='P0'))
 for name,family,formula,status in [
  ('risk_underwriting_price_surprise','pricing','actual_annual_interest_rate - crossfit_E(rate|income,amount,burden,bureau,employment,purpose)','FOLD_LOCAL'),
  ('risk_underwriting_price_surprise_scaled','pricing','pricing residual / training residual SD','FOLD_LOCAL'),
  ('riskq_joint_dpd_consumption','joint_dpd','mature90_DPD_survival_AUC * completed_order_spend30/observed_income3m','QUARANTINED_FOLD_LOCAL'),
  ('riskq_bureau_repayment_disagreement','bureau_dpd','actual_latest_provider_mean_score - crossfit_prediction(score|DPD,utilization,freshness,history_count)','QUARANTINED_FOLD_LOCAL'),
  ('riskq_recovery_strength_x_recent_dpd','recovery_dpd','(1-recovery_failure_3m)*recent15plus_DPD15_share','QUARANTINED_FOLD_LOCAL'),
  ('riskq_recovery_failure_x_recent_dpd','recovery_dpd','recovery_failure_3m*recent15plus_DPD15_share','QUARANTINED_FOLD_LOCAL')]:
  CATALOG.append(dict(feature_name=name,family=family,formula=formula,status=status,source='Application/account/bureau/repayment and audited V7 inputs',cutoff_rule='All regressions cross-fitted on correction rows; no default labels',leakage_risk='Unverified DPD snapshot' if name.startswith('riskq_') else 'Policy selection/confounding, incrementally low leakage',caveat='Pricing is policy generated; residual is association, not causal risk. DPD branches quarantined.',validation_status='Unvalidated',priority='P1'))
 for c in f.columns.difference(['application_id']):f[c]=pd.to_numeric(f[c]).replace([np.inf,-np.inf],np.nan)
 f.to_parquet(E/'data/raw_features.parquet',index=False);pd.DataFrame(CATALOG).to_csv(E/'reports/FEATURE_CATALOG.csv',index=False)
 dump(dict(status='READY',rows=len(f),columns=len(f.columns)-1,elapsed_seconds=time.time()-started,feature_sha256=sha(E/'data/raw_features.parquet'),
  no_default_labels=True,monthly_event_unit=True,complete_months_only=True,repayment_future_payments=0,
  observed_interest_rate_counts={str(k):int(v) for k,v in a.annual_interest_rate.value_counts().items()},
  blocked_exact_mechanism='DPD worsening before historical rejection cannot be dated from a single current snapshot. Observed allocation-event sequence is an explicitly different proxy.'),E/'reports/FEATURE_STATUS.json')
 print('Extension raw features READY',flush=True)

def fit_fold(fold):
 folder=E/'data'/f'fold{fold}';folder.mkdir(parents=True,exist_ok=True)
 raw=pd.read_parquet(E/'data/raw_features.parquet');base_train=pd.read_parquet(ROOT/'data/train_features.parquet');base_test=pd.read_parquet(ROOT/'data/test_features.parquet')
 idx=np.load(ROOT/'artifacts'/f'fold{fold}'/'indices.npz');hi,vi=idx['correction'],idx['validation'];assert not np.intersect1d(hi,vi).size
 parts={}
 for label,base in [('correction',base_train.iloc[hi]),('validation',base_train.iloc[vi]),('test',base_test)]:
  old=pd.read_parquet(V/'data'/f'fold{fold}'/(label+'.parquet'));d=combined(base,old)
  parts[label]=d.merge(raw,on='application_id',how='left',validate='one_to_one')
 h=parts['correction'];priors={}
 for kind,prefix in [('allocation','risk_'),('dpd','riskq_')]:
  post={};var={}
  for condition in ['negative','nonnegative']:
   k,n=[f'ext_aux_{kind}_{condition}_{v}' for v in ['k','n']];prior=beta_prior(h[k],h[n]);priors[kind+'_'+condition]=prior
   for label,d in parts.items():
    mean,variance=posterior(d[k],d[n],prior);d[prefix+'conditional_late_'+condition]=mean
    d[prefix+'conditional_'+condition+'_months']=d[n]
    post[(label,condition)]=mean;var[(label,condition)]=variance
  for label,d in parts.items():
   supported=(d[f'ext_aux_{kind}_negative_n']>0)&(d[f'ext_aux_{kind}_nonnegative_n']>0)
   suffix='dpd_cashflow_dependency' if kind=='dpd' else 'allocation_cashflow_dependency'
   d[prefix+suffix]=np.where(supported,post[(label,'negative')]-post[(label,'nonnegative')],np.nan)
   d[prefix+'conditional_dependency_sd']=np.where(supported,np.sqrt(var[(label,'negative')]+var[(label,'nonnegative')]),np.nan)
 models={}
 for kind,target in [('pricing','ext_aux_interest_rate'),('bureau_dpd','aux_bureau_score')]:
  def matrix(d):
   columns=['declared_monthly_income','loan_amount','aux_account_income','aux_bureau_score','aux_bureau_utilization','aux_account_months','ext_aux_contractual_active_scheduled','age','employment_tenure_months'] if kind=='pricing' else ['ext_aux_dpd_mean','ext_aux_recent_dpd15_share','aux_bureau_utilization','aux_bureau_freshness','aux_allocation_count']
   x=d[columns].to_numpy(float);x=np.sign(x)*np.log1p(abs(x))
   if kind=='pricing':
    for cat,levels in [('employment_type',['salaried','self_employed','variable_income']),('loan_purpose',['education','home_improvement','household','working_capital','medical'])]:
     x=np.column_stack([x,np.array([(d[cat]==v).to_numpy(float) for v in levels]).T])
   return x
  x=matrix(h);y=h[target].to_numpy(float);pred=np.full(len(h),np.nan);inner=[]
  for ti,qi in KFold(3,shuffle=True,random_state=31027+fold).split(x):
   assert not np.intersect1d(ti,qi).size
   m=train_nuisance(x[ti],y[ti],'bureau');pred[qi]=nuisance_predict(m,x[qi]);inner.append(dict(model=m,train_indices=ti,query_indices=qi))
  model=train_nuisance(x,y,'bureau');scale=max(float(np.nanstd(y-pred)),1e-8)
  for label,d in parts.items():
   p=pred if label=='correction' else nuisance_predict(model,matrix(d));res=d[target].to_numpy(float)-p
   if kind=='pricing':d['risk_underwriting_price_surprise']=res;d['risk_underwriting_price_surprise_scaled']=res/scale
   else:d['riskq_bureau_repayment_disagreement']=res
  models[kind]=dict(final=model,inner=inner,scale=scale)
 for d in parts.values():
  d['riskq_joint_dpd_consumption']=d.v7q_repay_dpd_severity_auc*div(d.aux_spend30,d.aux_account_income)
  d['riskq_recovery_strength_x_recent_dpd']=(1-d.risk_recovery_failure_3m)*d.ext_aux_recent_dpd15_share
  d['riskq_recovery_failure_x_recent_dpd']=d.risk_recovery_failure_3m*d.ext_aux_recent_dpd15_share
 for label,d in parts.items():
  keep=['application_id']+[c for c in d if c.startswith(('v7_','v7q_','aux_','ext_aux_','risk_','riskq_'))]
  d[keep].to_parquet(folder/(label+'.parquet'),index=False)
 joblib.dump(dict(priors=priors,nuisance=models,correction_training_ids=h.application_id.tolist()),folder/'nuisance.joblib',compress=3)
 dump(dict(status='PASS',fold=fold,correction_ids_sha256=ids_hash(base_train,hi),validation_ids_sha256=ids_hash(base_train,vi),
  Beta_Binomial_priors=priors,no_default_targets_used=True,outer_validation_and_test_not_fitted=True,inner_crossfit_splits=3,
  raw_sha256=sha(E/'data/raw_features.parquet')),folder/'AUDIT.json');print('Extension fold READY',fold,flush=True)

def checks():
 # Fully observed failure, successful recovery, and right-censored deficit are distinct.
 r=recovery_statistics(np.array([0,1,2,3]),np.array([-5,2,2,2]),np.array([10,5,7,9]),np.array([5,7,9,11]))
 assert r['time']==1 and r['failure']==0 and r['balance_failure']==0
 r=recovery_statistics(np.array([0,1,2,3]),np.array([-5,-1,-1,-1]),np.array([10,5,4,3]),np.array([5,4,3,2]))
 assert r['failure']==1 and r['balance_failure']==1 and r['three_observed']==1
 r=recovery_statistics(np.array([0,2,3]),np.array([-5,1,1]),np.array([10,5,6]),np.array([5,6,7]))
 assert np.isnan(r['probability']) and np.isnan(r['failure'])
 p=beta_prior([0,5],[1,10]);mean,var=posterior([0,5],[1,10],p);assert np.all((mean>0)&(mean<1)) and np.all(var>0)
 assert np.isnan(div(2,0)) and np.isnan(div(2,-1))
 (E/'reports').mkdir(parents=True,exist_ok=True);dump(dict(status='PASS',checks=['recovery_success','matured_failure','calendar_gap_and_right_censoring','posterior_support','no_epsilon_zero_capacity']),E/'reports/UNIT_CHECKS.json')
 print('Extension integrity checks PASS',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int);p.add_argument('--checks',action='store_true');args=p.parse_args()
 if args.checks:checks()
 elif args.fold is not None:fit_fold(args.fold)
 else:build()
