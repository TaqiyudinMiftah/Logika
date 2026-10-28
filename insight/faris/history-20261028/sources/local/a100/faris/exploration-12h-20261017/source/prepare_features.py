"""Target-free new temporal features and histories; runs inside the campaign clock."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,gc
from pathlib import Path
import numpy as np,pandas as pd
from utils import read,write,sha,ident

def build(raw,bank,out):
 raw=Path(raw);bank=Path(bank);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 meta=pd.read_parquet(bank/'metadata.parquet');customers=pd.Index(meta.customer_id);cut=meta[['customer_id','decision_datetime']].copy();cut.decision_datetime=pd.to_datetime(cut.decision_datetime)
 features={};info={};event_parts=[]
 def add(name,s,family,window='all',**extra):
  features[name]=pd.Series(s).reindex(customers).replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float32)
  source={'cross_source':['account','application','profile'],'shopping':['shopping'],'repayment':['repayment'],'bureau':['bureau'],'account':['account']}[family]
  info[name]={'family':family,'window':window,'sources':source,**extra}
 def hist(file,date):
  d=pd.read_csv(raw/(file+'.csv'));d[date]=pd.to_datetime(d[date]);d=d.merge(cut,on='customer_id',validate='many_to_one');assert d[date].lt(d.decision_datetime).all();d['age']=(d.decision_datetime-d[date]).dt.total_seconds()/86400;return d
 rep=hist('repayment_history','due_date');rep['fraction']=(rep.amount_paid/rep.amount_due.where(rep.amount_due>0)).clip(0,1);rep['late']=rep.max_days_past_due.ge(30).astype(float)
 b=hist('bureau_observations','available_datetime');b['report_datetime']=pd.to_datetime(b.report_datetime);assert b.report_datetime.lt(b.decision_datetime).all()
 acc=hist('account_behavior','month');assert (acc.month+pd.offsets.MonthBegin(1)<=acc.decision_datetime).all();acc['surplus']=acc.income_like_credits-acc.outflows
 e=hist('ecommerce_transactions','transaction_datetime');e['price']=e.item_total/e.quantity.where(e.quantity>0)
 for w in [30,90,180,365,'all']:
  rr=rep if w=='all' else rep[rep.age<=w];bb=b if w=='all' else b[b.age<=w];aa=acc if w=='all' else acc[acc.age<=w];ee=e if w=='all' else e[e.age<=w]
  prefix=f'new_{w}_';g=rr.groupby('customer_id');add(prefix+'exposure_unpaid',(rr.amount_due-rr.amount_paid).clip(lower=0).groupby(rr.customer_id).sum()/g.amount_due.sum().replace(0,np.nan),'repayment',w)
  ordered=rr.sort_values(['customer_id','due_date']);ordered['change']=ordered.groupby('customer_id').late.diff();add(prefix+'late_transitions',ordered.groupby('customer_id').change.mean(),'repayment',w)
  for half in [30,90,180]:
   weights=np.exp(-np.log(2)*rr.age/half);num=(weights*rr.late).groupby(rr.customer_id).sum();den=weights.groupby(rr.customer_id).sum();add(prefix+f'decay_{half}',num/den.replace(0,np.nan),'repayment',w,half_life=half)
  for t in [1,7,30,60,90]:
   eligible=rr[rr.age>=t];late=eligible.max_days_past_due.ge(t).groupby(eligible.customer_id).sum();count=eligible.groupby('customer_id').size();add(prefix+f'mature_{t}_k',late,'repayment',w);add(prefix+f'mature_{t}_n',count,'repayment',w)
  recent=rr[rr.age<=90].groupby('customer_id').late.mean();older=rr[rr.age>90].groupby('customer_id').late.mean();add(prefix+'recent_deterioration',recent-older,'repayment',w)
  latest=bb.sort_values('report_datetime').drop_duplicates(['customer_id','bureau_provider'],keep='last');pivot=latest.pivot(index='customer_id',columns='bureau_provider',values='score')
  add(prefix+'provider_disagreement',pivot.max(axis=1)-pivot.min(axis=1),'bureau',w);add(prefix+'provider_coverage',pivot.notna().sum(axis=1),'bureau',w)
  sortedb=bb.sort_values('report_datetime');changes=sortedb.groupby(['customer_id','bureau_provider']).score.diff();add(prefix+'score_change',changes.groupby(sortedb.customer_id).mean(),'bureau',w)
  add(prefix+'score_staleness_product',(bb.score*bb.age).groupby(bb.customer_id).mean(),'bureau',w)
  matched=bb.assign(report_bucket=bb.report_datetime.dt.to_period('M').astype(str)).groupby(['customer_id','report_bucket']).score.agg(['min','max']);add(prefix+'matched_provider_disagreement',(matched['max']-matched['min']).groupby('customer_id').mean(),'bureau',w)
  for col in ['surplus','income_like_credits','outflows','average_balance']:
   g=aa.groupby('customer_id')[col];add(prefix+col+'_cv',g.std()/g.mean().abs().replace(0,np.nan),'account',w)
   ordered=aa.sort_values('month');add(prefix+col+'_last_change',ordered.groupby('customer_id')[col].diff().groupby(ordered.customer_id).last(),'account',w)
  add(prefix+'persistent_deficit',aa.surplus.lt(0).groupby(aa.customer_id).mean(),'account',w);add(prefix+'balance_recovery',aa.sort_values('month').groupby('customer_id').average_balance.diff().gt(0).groupby(aa.sort_values('month').customer_id).mean(),'account',w)
  orders=ee.drop_duplicates('order_id');paid=orders[orders.order_status.eq('completed')];cancel=orders[orders.order_status.eq('cancelled')]
  add(prefix+'cancel_basket_mean',cancel.groupby('customer_id').order_total.mean(),'shopping',w);add(prefix+'cancel_rate',orders.order_status.eq('cancelled').groupby(orders.customer_id).mean(),'shopping',w)
  intervals=paid.sort_values('transaction_datetime').groupby('customer_id').transaction_datetime.diff().dt.total_seconds()/86400
  add(prefix+'purchase_interval_cv',intervals.groupby(paid.sort_values('transaction_datetime').customer_id).std()/intervals.groupby(paid.sort_values('transaction_datetime').customer_id).mean().replace(0,np.nan),'shopping',w)
  cats=ee[ee.order_status.eq('completed')].groupby(['customer_id','product_category']).price.median().unstack();center=np.exp(np.log(cats.where(cats>0)).mean(axis=1))
  for col in cats:add(prefix+'relative_'+str(col),cats[col]/center,'shopping',w)
  pay=pd.crosstab(paid.customer_id,paid.payment_method);shares=pay.div(pay.sum(axis=1),axis=0);add(prefix+'payment_entropy',-(shares.where(shares>0)*np.log(shares.where(shares>0))).sum(axis=1),'shopping',w)
  print('Feature window',w,flush=True)
 # Affordability from observed histories; no label or future realization is used.
 train=pd.read_csv(raw/'train_applications.csv').drop(columns='target');test=pd.read_csv(raw/'test_applications.csv');apps=pd.concat([train,test]);profiles=pd.read_csv(raw/'customer_profiles.csv');apps=apps.merge(profiles,on='customer_id',validate='one_to_one').set_index('customer_id')
 income=acc.groupby('customer_id').income_like_credits.mean();spend=acc.groupby('customer_id').outflows.mean();decl=apps.declared_monthly_income;inst=apps.proposed_monthly_installment
 for name,val in {'income_discrepancy':(income-decl)/(decl.abs()+1),'disposable_coverage':(income-spend)/(inst+1),'installment_income':inst/(income+1),'income_source_reliability':acc.groupby('customer_id').size()/(1+acc.groupby('customer_id').income_like_credits.std()/(income.abs()+1))}.items():add('new_cross_'+name,val,'cross_source')
 # Unified history stream: source flags, age, and source-specific amounts. Only eligible records.
 for source,d,vals in [('repayment',rep,['amount_due','amount_paid','max_days_past_due']),('bureau',b,['score','utilization','outstanding_balance']),('account',acc,['income_like_credits','outflows','average_balance']),('shopping',e.drop_duplicates('order_id'),['order_total'])]:
  ids=customers.get_indexer(d.customer_id);z=np.zeros((len(d),9),dtype=np.float32);z[:,0]=np.log1p(d.age);z[:,1+['repayment','bureau','account','shopping'].index(source)]=1
  for j,col in enumerate(vals):z[:,5+j]=np.sign(d[col])*np.log1p(np.abs(d[col]))
  if source=='shopping':z[:,8]=d.order_status.eq('cancelled').astype(float)
  event_parts.append((ids,d.age.to_numpy(),z))
 ids=np.concatenate([v[0] for v in event_parts]);age=np.concatenate([v[1] for v in event_parts]);events=np.concatenate([v[2] for v in event_parts]);assert (ids>=0).all()
 seq=np.lib.format.open_memmap(out/'events.npy',mode='w+',dtype='float32',shape=(len(meta),128,9));seq[:]=0;lengths=np.zeros(len(meta),np.int16)
 order=np.lexsort((age,ids));last=-1;rank=0
 for i in order:
  row=ids[i]
  if row!=last:last=row;rank=0
  if rank<128:seq[row,rank]=np.nan_to_num(events[i]);rank+=1;lengths[row]=rank
 seq.flush();np.save(out/'lengths.npy',lengths)
 names=list(features);matrix=np.lib.format.open_memmap(out/'numeric.npy',mode='w+',dtype='float32',shape=(len(meta),len(names)))
 for j,n in enumerate(names):matrix[:,j]=features[n]
 matrix.flush();manifest={'columns':names,'info':info,'shape':[len(meta),len(names)],'base_bank_id':read(bank/'manifest.json')['bank_id'],'rows_sha256':ident(meta.application_id.tolist()),'files':{n:sha(out/n) for n in ['numeric.npy','events.npy','lengths.npy']},'policy':'No labels; decision-time history only; completed spending distinct from cancelled baskets'};write(out/'manifest.json',manifest)
 print('Prepared extras',manifest['shape'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('raw');p.add_argument('bank');p.add_argument('output');a=p.parse_args();build(a.raw,a.bank,a.output)
