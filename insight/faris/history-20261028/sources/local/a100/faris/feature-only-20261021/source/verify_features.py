"""Independent scalar/raw-row reconstruction of all18 features for24 applicants."""
from pathlib import Path
import numpy as np
import pandas as pd
import baseline as b
from features import BLOCKS
R=Path(__file__).resolve().parents[1];RAW=R.parent/'exploration-12h-20261017/raw'
def main():
 f=pd.read_parquet(R/'features.parquet').set_index('customer_id')
 # Deterministic choice based on input order only, no labels or performance.
 ids=f.index[np.random.default_rng(20261024).choice(len(f),24,replace=False)]
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['customer_id','decision_datetime'])for s in ['train','test']]).set_index('customer_id')
 tables={}
 for name in ['account_behavior','repayment_history','bureau_observations','ecommerce_transactions']:
  x=pd.read_csv(RAW/(name+'.csv'));tables[name]=x[x.customer_id.isin(ids)].copy()
 for name,cols in [('account_behavior',['month']),('repayment_history',['due_date','last_payment_datetime']),('bureau_observations',['report_datetime','available_datetime']),('ecommerce_transactions',['transaction_datetime'])]:
  for c in cols:tables[name][c]=pd.to_datetime(tables[name][c])
 errors=[]
 def divide(a,d):return a/d if d>0 else np.nan
 def mean(x):return float(np.mean(x))if len(x)else np.nan
 for cid in ids:
  cutoff=pd.Timestamp(apps.loc[cid,'decision_datetime']);actual={}
  ac=tables['account_behavior'].query('customer_id==@cid').copy();ac=ac[ac.month+pd.offsets.MonthBegin(1)<=cutoff].set_index('month')
  rp=tables['repayment_history'].query('customer_id==@cid').copy();rp=rp[((cutoff-rp.due_date).dt.total_seconds()/86400>=30)&rp.amount_due.gt(0)]
  due_total=float(rp.amount_due.sum());monthly=[]
  for period,x in rp.groupby(rp.due_date.dt.to_period('M')):
   month=period.to_timestamp()
   if month not in ac.index:continue
   row=ac.loc[month];amount=float(x.amount_due.sum());allocated=float(x.loc[x.last_payment_datetime.notna()&x.last_payment_datetime.lt(cutoff),'amount_due'].sum())
   monthly.append(dict(due=amount,allocated=allocated,negative=row.inflows<row.outflows,income=row.income_like_credits,recent=month>=cutoff.to_period('M').to_timestamp()-pd.DateOffset(months=3)))
  due=sum(x['due']for x in monthly)
  actual['fe_due_income_mean']=mean([divide(x['due'],x['income'])for x in monthly if x['income']>0])
  actual['fe_negative_cashflow_due_share']=divide(sum(x['due']for x in monthly if x['negative']),due)
  actual['fe_due_account_overlap_share']=divide(due,due_total)if monthly else np.nan
  for label,negative in [('negative',True),('nonnegative',False)]:
   subset=[x for x in monthly if x['negative']==negative]
   actual['fe_'+label+'_allocated_due_share']=divide(sum(x['allocated']for x in subset),sum(x['due']for x in subset))
  ss=[]
  for recent in [True,False]:
   subset=[x for x in monthly if x['recent']==recent];ss.append(divide(sum(x['due']for x in subset if x['negative']),sum(x['due']for x in subset)))
  actual['fe_recent_negative_due_share_change']=ss[0]-ss[1]
  bu=tables['bureau_observations'].query('customer_id==@cid');bu=bu[bu.available_datetime.lt(cutoff)&bu.report_datetime.lt(cutoff)]
  slopes=[];debt=[];ages=[];lags=[]
  for provider,x in bu.groupby('bureau_provider'):
   x=x.sort_values('available_datetime').drop_duplicates('report_datetime',keep='last').sort_values(['report_datetime','available_datetime'])
   first,last=x.iloc[0],x.iloc[-1];days=(last.report_datetime-first.report_datetime).total_seconds()/86400
   ages.append((cutoff-last.report_datetime).total_seconds()/86400);lags.append((last.available_datetime-last.report_datetime).total_seconds()/86400)
   if days>=90:
    if pd.notna(first.score)and pd.notna(last.score):slopes.append(365*(last.score-first.score)/days)
    if first.outstanding_balance>=0 and last.outstanding_balance>=0:debt.append(365*np.log((1+last.outstanding_balance)/(1+first.outstanding_balance))/days)
  actual.update(fe_provider_score_slope_mean=mean(slopes),fe_provider_score_slope_min=min(slopes)if slopes else np.nan,fe_provider_score_slope_disagreement=float(np.std(slopes,ddof=1))if len(slopes)>1 else np.nan,fe_provider_logdebt_slope_mean=mean(debt),fe_latest_report_age_median=float(np.median(ages))if ages else np.nan,fe_latest_reporting_lag_median=float(np.median(lags))if lags else np.nan)
  ec=tables['ecommerce_transactions'].query('customer_id==@cid').copy();age=(cutoff-ec.transaction_datetime).dt.total_seconds()/86400;ec=ec[age.gt(0)&age.le(365)].copy();ec['recent']=age.loc[ec.index].le(90)
  maps=[];counts=[];cancel=[];gaps=[]
  for recent in [True,False]:
   part=ec[ec.recent.eq(recent)];completed=part[part.order_status.eq('completed')&part.quantity.gt(0)&part.item_total.ge(0)]
   maps.append({cat:float(np.log1p(x.item_total/x.quantity).mean())for cat,x in completed.groupby('product_category')})
   counts.append(completed.product_category.value_counts().to_dict());orders=part.drop_duplicates('order_id')
   cancel.append(float(orders.order_status.eq('cancelled').mean())if len(orders)else np.nan)
   times=orders[orders.order_status.eq('completed')].transaction_datetime.sort_values();delta=np.diff(times.astype('int64'))/86400e9;gaps.append(float(np.median(delta))if len(delta)else np.nan)
  changes=[maps[0][cat]-maps[1][cat]for cat in set(maps[0])&set(maps[1])]
  n0,n1=sum(counts[0].values()),sum(counts[1].values());turnover=.5*sum(abs(counts[0].get(cat,0)/n0-counts[1].get(cat,0)/n1)for cat in set(counts[0])|set(counts[1]))if n0 and n1 else np.nan
  actual.update(fe_matched_category_logprice_change=mean(changes),fe_matched_category_change_SE=float(np.std(changes,ddof=1)/np.sqrt(len(changes)))if len(changes)>1 else np.nan,fe_matched_category_count=len(changes),fe_category_mix_turnover=turnover,fe_cancellation_share_change=cancel[0]-cancel[1],fe_purchase_gap_logratio=np.log1p(gaps[0])-np.log1p(gaps[1]))
  assert set(actual)==set(sum(BLOCKS.values(),[]))
  for c,v in actual.items():
   saved=float(f.loc[cid,c]);assert np.isclose(v,saved,rtol=1e-10,atol=1e-10,equal_nan=True),(cid,c,v,saved)
   if np.isfinite(v):errors.append(abs(v-saved))
 b.dump(dict(status='passed',applicants=24,features=18,scalar_comparisons=432,max_abs_error=max(errors),target_labels_opened=False,selected_without_labels=True,feature_hash=b.sha(R/'features.parquet')),R/'INDEPENDENT_FEATURE_AUDIT.json')
if __name__=='__main__':main()
