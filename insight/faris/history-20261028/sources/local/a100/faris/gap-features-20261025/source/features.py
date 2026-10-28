"""Row-local features from explicitly pre-decision records; no target fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import argparse,datetime,hashlib,json,time
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
RAW=Path('/mnt/home/logika-faris-exploration/exploration-12h-20261017/raw')
FAMILIES={
 'liquidity':['liq_shortfall_fraction','liq_shortfall_income_mean','liq_lowbuffer_fraction',
              'liq_surplus_income_p10','liq_joint_stress_fraction','liq_longest_shortfall_run_fraction',
              'liq_recent_shortfall_change','liq_observed_months'],
 'alignment':['align_spend_outflow_share','align_spend_income_correlation',
              'align_lowincome_spend_logratio','align_lowbuffer_cancel_change',
              'align_lowbuffer_credit_change','align_negativeflow_spend_share',
              'align_spend_outflow_correlation','align_observed_months'],
 'application_path':['path_reapply_after_nonapproval_fraction','path_current_last_request_logratio',
                     'path_trailing_nonapproved_count','path_last_gap_relative_log',
                     'path_nonapproved_request_logchange','path_recent_nonapproved_request_income',
                     'path_approved_haircut_mean','path_days_since_nonapproved']}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()

def load():
 cols=['application_id','customer_id','decision_datetime','loan_amount','proposed_monthly_installment']
 apps=pd.concat([pd.read_csv(RAW/f'{kind}_applications.csv',usecols=cols)for kind in ['train','test']],ignore_index=True)
 apps['decision_datetime']=pd.to_datetime(apps.decision_datetime)
 profiles=pd.read_csv(RAW/'customer_profiles.csv',usecols=['customer_id','declared_monthly_income'])
 apps=apps.merge(profiles,on='customer_id',validate='one_to_one')
 assert len(apps)==100000 and apps.application_id.is_unique and apps.customer_id.is_unique
 account=pd.read_csv(RAW/'account_behavior.csv',parse_dates=['month'])
 columns=['order_id','customer_id','transaction_datetime','order_total','order_status','payment_method']
 # Exact order deduplication before any monetary sum, including chunk boundaries.
 orders=pd.read_csv(RAW/'ecommerce_transactions.csv',usecols=columns)
 first=orders.drop_duplicates('order_id')
 for col in columns[1:]:
  assert orders.groupby('order_id',sort=False)[col].nunique(dropna=False).max()==1,col
 orders=first.copy();orders['transaction_datetime']=pd.to_datetime(orders.transaction_datetime)
 previous=pd.read_csv(RAW/'previous_applications.csv',parse_dates=['application_datetime','decision_datetime'])
 return apps,account,orders,previous

def bounded(apps,account,orders,previous):
 lookup=apps.set_index('customer_id').decision_datetime
 account=account[account.customer_id.isin(lookup.index)].copy()
 account['cutoff']=account.customer_id.map(lookup)
 account=account[(account.month+pd.offsets.MonthBegin(1)<=account.cutoff)&
                 (account.month>=account.cutoff.dt.to_period('M').dt.to_timestamp()-pd.DateOffset(months=12))].copy()
 assert not account.duplicated(['customer_id','month']).any()
 orders=orders[orders.customer_id.isin(lookup.index)].copy();orders['cutoff']=orders.customer_id.map(lookup)
 orders=orders[orders.transaction_datetime<orders.cutoff].copy()
 assert not orders.order_id.duplicated().any()
 previous=previous[previous.customer_id.isin(lookup.index)].copy();previous['cutoff']=previous.customer_id.map(lookup)
 previous=previous[(previous.decision_datetime<previous.cutoff)&(previous.application_datetime<previous.cutoff)].copy()
 return account,orders,previous

def corr(x,y):
 if len(x)<4 or np.std(x)<1e-9 or np.std(y)<1e-9:return np.nan
 return float(np.corrcoef(x,y)[0,1])

def liquidity(g,inst):
 g=g.sort_values('month');income=g.income_like_credits.to_numpy(float)
 net=(g.inflows-g.outflows).to_numpy(float);low=g.minimum_balance.to_numpy(float)<inst
 short=net<inst;n=len(g);valid=income>0
 ratio=np.divide(net-inst,income,out=np.full(n,np.nan),where=valid)
 months=(g.month.dt.year*12+g.month.dt.month).to_numpy()
 run=0;longest=0
 for j,x in enumerate(short):
  if x:run=run+1 if j and months[j]-months[j-1]==1 else 1
  else:run=0
  longest=max(longest,run)
 recent=g.month>=g.cutoff.dt.to_period('M').dt.to_timestamp()-pd.DateOffset(months=3)
 change=float(short[recent].mean()-short[~recent].mean())if recent.any()and(~recent).any()else np.nan
 return [float(short.mean()),float(np.nanmean(np.maximum(-ratio,0))),float(low.mean()),
         float(np.nanquantile(ratio,.1)),float(np.mean(short&(g.closing_balance.to_numpy()<3*inst))),
         longest/n,change,n]

def alignment(g,inst,shuffle=False):
 g=g.sort_values('month');n=len(g)
 if n<3:return [np.nan]*7+[n]
 x=g[['spend','cancel','basket','credit']].to_numpy(float)
 if shuffle:
  # Stable content-based permutation, independent of targets and source ordering.
  seed=int(hashlib.sha256(str(g.customer_id.iloc[0]).encode()).hexdigest()[:8],16)^2026102517
  x=x[np.random.default_rng(seed).permutation(n)]
 spend,cancel,basket,credit=x.T
 income=g.income_like_credits.to_numpy(float);out=g.outflows.to_numpy(float)
 lowincome=income<np.median(income);lowbuffer=g.minimum_balance.to_numpy(float)<inst
 def split_ratio(numerator,denominator,mask):
  vals=[]
  for keep in (mask,~mask):
   d=denominator[keep].sum();vals.append(numerator[keep].sum()/d if d>0 else np.nan)
  return vals[0]-vals[1]
 logratio=np.log1p(spend[lowincome].mean())-np.log1p(spend[~lowincome].mean())if lowincome.any()and(~lowincome).any()else np.nan
 return [spend.sum()/out.sum()if out.sum()>0 else np.nan,corr(np.log1p(spend),np.log1p(income)),
         logratio,split_ratio(cancel,basket,lowbuffer),split_ratio(credit,spend,lowbuffer),
         spend[(g.inflows-g.outflows).to_numpy()<0].sum()/spend.sum()if spend.sum()>0 else np.nan,
         corr(np.log1p(spend),np.log1p(out)),n]

def path_features(g,app):
 g=g.sort_values(['decision_datetime','application_datetime','application_id'])
 non=g.decision.ne('approved').to_numpy();n=len(g)
 gaps=g.decision_datetime.diff().dt.total_seconds().to_numpy()/86400
 followups=(g.application_datetime.iloc[1:].to_numpy()-g.decision_datetime.iloc[:-1].to_numpy())/np.timedelta64(1,'D')
 lastgap=(app.decision_datetime-g.decision_datetime.iloc[-1]).total_seconds()/86400
 trailing=0
 for value in non[::-1]:
  if not value:break
  trailing+=1
 req=g.requested_amount.to_numpy(float);changes=np.diff(np.log1p(req))
 mask=non[:-1]
 recent=((app.decision_datetime-g.decision_datetime).dt.total_seconds()/86400<=90)&non
 ap=g[g.decision.eq('approved')&g.approved_amount.notna()&g.requested_amount.gt(0)]
 non_dates=g.loc[non,'decision_datetime']
 return [float(np.mean((followups>=0)&(followups<=90)&mask))if n>1 else np.nan,
         float(np.log1p(app.loan_amount)-np.log1p(req[-1])),trailing,
         float(np.log1p(lastgap)-np.log1p(np.median(gaps[1:])))if n>1 else np.nan,
         float(changes[mask].mean())if mask.any()else np.nan,
         float(g.loc[recent,'requested_amount'].sum()/max(app.declared_monthly_income,1)),
         float(np.log((ap.approved_amount+1)/(ap.requested_amount+1)).mean())if len(ap)else np.nan,
         (app.decision_datetime-non_dates.max()).total_seconds()/86400 if len(non_dates)else np.nan]

def build(apps,account,orders,previous):
 account,orders,previous=bounded(apps,account,orders,previous)
 table=apps.set_index('customer_id');cols=sum(FAMILIES.values(),[])
 features=pd.DataFrame(np.nan,index=table.index,columns=cols)
 null=pd.DataFrame(np.nan,index=table.index,columns=FAMILIES['alignment'])
 for cid,g in account.groupby('customer_id',sort=False):
  features.loc[cid,FAMILIES['liquidity']]=liquidity(g,float(table.loc[cid,'proposed_monthly_installment']))
 orders['month']=orders.transaction_datetime.dt.to_period('M').dt.to_timestamp()
 orders['spend']=orders.order_total.where(orders.order_status.eq('completed'),0)
 orders['cancel']=orders.order_total.where(orders.order_status.eq('cancelled'),0)
 orders['credit']=orders.spend.where(orders.payment_method.eq('credit_card'),0)
 monthly=orders.groupby(['customer_id','month'],sort=False).agg(spend=('spend','sum'),cancel=('cancel','sum'),basket=('order_total','sum'),credit=('credit','sum')).reset_index()
 first=orders.groupby('customer_id').month.min()
 matched=account.merge(monthly,on=['customer_id','month'],how='left',validate='one_to_one')
 # Recorded-spending zeroes only after the first observed shopping month.
 # This is a coverage proxy, not evidence of an enrollment date or total spending.
 matched=matched[matched.month>matched.customer_id.map(first)].copy()
 matched[['spend','cancel','basket','credit']]=matched[['spend','cancel','basket','credit']].fillna(0)
 for cid,g in matched.groupby('customer_id',sort=False):
  inst=float(table.loc[cid,'proposed_monthly_installment'])
  features.loc[cid,FAMILIES['alignment']]=alignment(g,inst)
  null.loc[cid]=alignment(g,inst,True)
 for cid,g in previous.groupby('customer_id',sort=False):features.loc[cid,FAMILIES['application_path']]=path_features(g,table.loc[cid])
 features=features.replace([np.inf,-np.inf],np.nan);null=null.replace([np.inf,-np.inf],np.nan)
 for count in ['liq_observed_months','align_observed_months']:features[count]=features[count].fillna(0)
 null['align_observed_months']=null.align_observed_months.fillna(0)
 features.insert(0,'application_id',table.application_id);null.insert(0,'application_id',table.application_id)
 return features.reset_index(drop=True),null.reset_index(drop=True)

def main():
 started=time.time();out=R/'features';out.mkdir(exist_ok=False)
 inputs=load();apps,account,orders,previous=inputs
 features,null=build(*inputs)
 features.to_parquet(out/'true.parquet',index=False);null.to_parquet(out/'alignment_null.parquet',index=False)
 # Invariance checks use deterministic development-only customers, without labels.
 chosen=apps.iloc[:80];ids=set(chosen.customer_id)
 parts=[x[x.customer_id.isin(ids)].copy()for x in inputs[1:]]
 expected=features.iloc[:80].set_index('application_id')
 reordered=build(chosen.sample(frac=1,random_state=71),*[x.sample(frac=1,random_state=72)for x in parts])[0].set_index('application_id').loc[expected.index]
 assert np.allclose(expected,reordered,rtol=1e-12,atol=1e-10,equal_nan=True)
 # Explicit future records must be ignored. Preserve keys but shift every time.
 future=[]
 for x,timecols in zip(parts,[['month'],['transaction_datetime'],['application_datetime','decision_datetime']]):
  y=x.copy()
  for col in timecols:y[col]=pd.Timestamp('2099-01-01')
  if 'order_id'in y:y['order_id']=y.order_id+'__future'
  future.append(pd.concat([x,y],ignore_index=True))
 poisoned=build(chosen,*future)[0].set_index('application_id').loc[expected.index]
 assert np.allclose(expected,poisoned,rtol=1e-12,atol=1e-10,equal_nan=True)
 finite_counts=features.drop(columns='application_id').notna().sum().to_dict()
 audit={'status':'passed','source_sha256':sha(__file__),'target_labels_used':False,'rows':len(features),
        'future_record_invariance':True,'input_order_invariance':True,'sample_check_customers':80,
        'raw_rows':{'account':len(account),'deduplicated_orders':len(orders),'previous':len(previous)},
        'families':FAMILIES,'finite_counts':{k:int(v)for k,v in finite_counts.items()},
        'files':{p.name:sha(p)for p in out.glob('*.parquet')},
        'raw_sha256':{name:sha(RAW/name)for name in ['train_applications.csv','test_applications.csv','customer_profiles.csv','account_behavior.csv','ecommerce_transactions.csv','previous_applications.csv']},
        'seconds':time.time()-started,'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_boundary':'New fields read no repayment values. All records filtered before decision; account months complete and last12months only.'}
 (out/'GATE.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
