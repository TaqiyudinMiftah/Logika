"""Applicant-local financial features. No outcome column is ever read here."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import baseline as b

R=Path(__file__).resolve().parents[1]
RAW=R.parent/'exploration-12h-20261017/raw'
BLOCKS={
 'repayment_cashflow': ['fe_due_income_mean','fe_negative_cashflow_due_share','fe_due_account_overlap_share','fe_negative_allocated_due_share','fe_nonnegative_allocated_due_share','fe_recent_negative_due_share_change'],
 'bureau_trajectory': ['fe_provider_score_slope_mean','fe_provider_score_slope_min','fe_provider_logdebt_slope_mean','fe_provider_score_slope_disagreement','fe_latest_report_age_median','fe_latest_reporting_lag_median'],
 'shopping_change': ['fe_matched_category_logprice_change','fe_matched_category_change_SE','fe_matched_category_count','fe_category_mix_turnover','fe_cancellation_share_change','fe_purchase_gap_logratio'],
}

def build():
 start=time.time();assert not (R/'FEATURE_AUDIT.json').exists()
 paths=[RAW/(s+'_applications.csv')for s in ['train','test']]
 apps=pd.concat([pd.read_csv(p,usecols=['application_id','customer_id','decision_datetime'])for p in paths],ignore_index=True)
 assert len(apps)==100000 and apps.application_id.is_unique and apps.customer_id.is_unique
 apps.decision_datetime=pd.to_datetime(apps.decision_datetime);apps['row']=np.arange(len(apps));n=len(apps)
 out=apps[['application_id','customer_id']].copy();aud={}
 def hist(name,dates,columns=None):
  p=RAW/(name+'.csv');paths.append(p)
  x=pd.read_csv(p,usecols=columns,parse_dates=dates)
  assert 'target'not in x
  return x.merge(apps[['customer_id','decision_datetime','row']],on='customer_id',validate='many_to_one')
 def put(name,s,empty=np.nan):out[name]=s.reindex(range(n),fill_value=empty).to_numpy(float)
 def ratio(a,c):return a/c.where(c.gt(0))
 ac=hist('account_behavior',['month'])
 eligible=ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime
 aud['account_ineligible_rows']=int((~eligible).sum());ac=ac[eligible].copy()
 assert not ac.duplicated(['row','month']).any()
 ac['net']=ac.inflows-ac.outflows
 ac['identity_residual']=ac.closing_balance-ac.opening_balance-ac.inflows+ac.outflows
 aud['account_balance_identity_nonzero_fraction']=float(ac.identity_residual.abs().gt(1).mean())
 rp=hist('repayment_history',['due_date','last_payment_datetime'])
 assert not rp.duplicated(['row','installment_id']).any()
 age=(rp.decision_datetime-rp.due_date).dt.total_seconds()/86400
 aud['repayment_future_due_rows']=int((age<=0).sum())
 eligible=(age>=30)&rp.amount_due.gt(0)
 rp=rp[eligible].copy();rp['allocated']=rp.last_payment_datetime.notna()&(rp.last_payment_datetime<rp.decision_datetime)
 aud['repayment_future_allocation_rows']=int((rp.last_payment_datetime>=rp.decision_datetime).sum())
 # Allocation presence is an observation indicator, not full or timely settlement.
 rp['allocated_due']=rp.amount_due*rp.allocated
 rp['month']=rp.due_date.dt.to_period('M').dt.to_timestamp()
 monthly=rp.groupby(['row','month']).agg(due=('amount_due','sum'),allocated_due=('allocated_due','sum')).reset_index()
 total=monthly.groupby('row').due.sum()
 monthly=monthly.merge(ac[['row','month','net','income_like_credits','decision_datetime']],on=['row','month'],validate='one_to_one')
 monthly['due_income']=ratio(monthly.due,monthly.income_like_credits)
 put('fe_due_income_mean',monthly.groupby('row').due_income.mean())
 neg=monthly.net.lt(0)
 put('fe_negative_cashflow_due_share',ratio(monthly[neg].groupby('row').due.sum().reindex(total.index,fill_value=0),monthly.groupby('row').due.sum()))
 put('fe_due_account_overlap_share',ratio(monthly.groupby('row').due.sum(),total))
 for label,mask in [('negative',neg),('nonnegative',~neg)]:
  g=monthly[mask].groupby('row')[['due','allocated_due']].sum()
  put('fe_'+label+'_allocated_due_share',ratio(g.allocated_due,g.due))
 recent=monthly.month>=monthly.decision_datetime.dt.to_period('M').dt.to_timestamp()-pd.DateOffset(months=3)
 shares=[]
 for mask in [recent,~recent]:
  denom=monthly[mask].groupby('row').due.sum();num=monthly[mask&neg].groupby('row').due.sum().reindex(denom.index,fill_value=0)
  shares.append(ratio(num,denom))
 put('fe_recent_negative_due_share_change',shares[0]-shares[1])
 aud['repayment_eligible_installments']=len(rp);aud['repayment_overlap_months']=len(monthly)
 del rp,monthly,ac

 bu=hist('bureau_observations',['report_datetime','available_datetime'])
 valid=(bu.available_datetime<bu.decision_datetime)&(bu.report_datetime<bu.decision_datetime)
 aud['bureau_ineligible_rows']=int((~valid).sum());bu=bu[valid].copy()
 assert (bu.available_datetime>=bu.report_datetime).all()
 # Use the last available version of a provider/report pair; never sum provider debts.
 bu=bu.sort_values('available_datetime').drop_duplicates(['row','bureau_provider','report_datetime'],keep='last')
 bu=bu.sort_values(['row','bureau_provider','report_datetime','available_datetime'])
 group=['row','bureau_provider'];first=bu.groupby(group).head(1).set_index(group);last=bu.groupby(group).tail(1).set_index(group)
 assert first.index.equals(last.index)
 span=(last.report_datetime-first.report_datetime).dt.total_seconds()/86400
 valid=span.ge(90)&first.score.notna()&last.score.notna()
 slope=((last.score-first.score)*365/span.where(valid)).where(valid)
 put('fe_provider_score_slope_mean',slope.groupby(level='row').mean())
 put('fe_provider_score_slope_min',slope.groupby(level='row').min())
 put('fe_provider_score_slope_disagreement',slope.groupby(level='row').std())
 debtvalid=span.ge(90)&first.outstanding_balance.ge(0)&last.outstanding_balance.ge(0)
 debt=((np.log1p(last.outstanding_balance)-np.log1p(first.outstanding_balance))*365/span.where(debtvalid)).where(debtvalid)
 put('fe_provider_logdebt_slope_mean',debt.groupby(level='row').mean())
 put('fe_latest_report_age_median',((last.decision_datetime-last.report_datetime).dt.total_seconds()/86400).groupby(level='row').median())
 put('fe_latest_reporting_lag_median',((last.available_datetime-last.report_datetime).dt.total_seconds()/86400).groupby(level='row').median())
 aud['bureau_providers_with_90day_pairs']=int(valid.sum());del bu,first,last

 ec=hist('ecommerce_transactions',['transaction_datetime'])
 age=(ec.decision_datetime-ec.transaction_datetime).dt.total_seconds()/86400
 aud['shopping_ineligible_rows']=int((age<=0).sum());ec=ec[age.gt(0)&age.le(365)].copy();ec['recent']=age.loc[ec.index].le(90)
 assert ec.groupby(['row','order_id']).order_status.nunique().max()==1
 orders=ec.drop_duplicates(['row','order_id']).copy()
 catmeans=[];catcounts=[];cancel=[];gaps=[]
 for recent in [True,False]:
  part=ec[ec.recent.eq(recent)&ec.order_status.eq('completed')&ec.quantity.gt(0)&ec.item_total.ge(0)].copy()
  part['logprice']=np.log1p(part.item_total/part.quantity)
  catmeans.append(part.groupby(['row','product_category']).logprice.mean())
  catcounts.append(part.groupby(['row','product_category']).size().unstack(fill_value=0))
  op=orders[orders.recent.eq(recent)]
  cancel.append(op.order_status.eq('cancelled').groupby(op.row).mean())
  op=op[op.order_status.eq('completed')].sort_values(['row','transaction_datetime'])
  delta=op.groupby('row').transaction_datetime.diff().dt.total_seconds()/86400
  gaps.append(delta.groupby(op.row).median())
 change=catmeans[0]-catmeans[1];cg=change.dropna().groupby(level='row')
 put('fe_matched_category_logprice_change',cg.mean())
 put('fe_matched_category_change_SE',cg.std()/np.sqrt(cg.count()))
 put('fe_matched_category_count',cg.count(),0)
 c0,c1=catcounts[0].align(catcounts[1],join='outer',fill_value=0)
 s0=c0.div(c0.sum(axis=1).replace(0,np.nan),axis=0);s1=c1.div(c1.sum(axis=1).replace(0,np.nan),axis=0)
 put('fe_category_mix_turnover',.5*(s0-s1).abs().sum(axis=1,min_count=1))
 put('fe_cancellation_share_change',cancel[0]-cancel[1])
 put('fe_purchase_gap_logratio',np.log1p(gaps[0])-np.log1p(gaps[1]))
 aud['shopping_matching_category_pairs']=int(change.notna().sum())
 columns=sum(BLOCKS.values(),[]);assert set(out)-{'application_id','customer_id'}==set(columns)
 out[columns]=out[columns].replace([np.inf,-np.inf],np.nan)
 assert out.application_id.tolist()==apps.application_id.tolist()
 out.to_parquet(R/'features.parquet',index=False)
 b.dump(dict(status='complete',rows=n,target_labels_opened=False,feature_shared_estimators=False,blocks=BLOCKS,
  audit=aud,source_hashes={str(p):b.sha(p)for p in paths},feature_hash=b.sha(R/'features.parquet'),
  distributions={c:dict(nonmissing=int(out[c].notna().sum()),unique=int(out[c].nunique()),quantiles=out[c].dropna().quantile([0,.1,.5,.9,1]).tolist())for c in columns},
  seconds=time.time()-start,qualification='Snapshot inherited; allocation availability is not proof of full settlement. Monthly net includes recorded outflows, not an independently observed debt-free surplus. No repayment amount_paid or DPD used.'),R/'FEATURE_AUDIT.json')
 print(json.dumps(dict(status='feature_audit_complete',seconds=time.time()-start,audit=aud)),flush=True)

if __name__=='__main__':build()
