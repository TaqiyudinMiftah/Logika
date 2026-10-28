"""Basket/event/pricing/maturity discoveries. All default labels excluded.

Population basket references are both fold-local and dated before the query
decision month. Own customer is removed from correction-training references.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import sys,json,time,argparse,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from features import ROOT,V,RAW,CATS,div
from baseline import dump,sha,ids_hash
from experiments import combined

S=V/'discovery';CATALOG=[];DAY=86400e9;TIERS=np.array([.16,.20,.24,.28,.32])

def add(f,name,value,family,formula,caveat=''):
 f[name]=np.asarray(value)
 CATALOG.append(dict(feature_name=name,family=family,formula=formula,status='IMPLEMENTED',source='Raw point-in-time application, previous, ecommerce, repayment, bureau tables',
  cutoff_rule='Only events before selected decision; own-history comparison uses strictly earlier windows',leakage_risk='Incrementally low; inherited baseline uncertainty remains',
  caveat=caveat,validation_status='Unvalidated',priority='P1'))

def schedule_end(orig,terms):
 number=orig.dt.year*12+orig.dt.month-1+terms
 first=pd.to_datetime(pd.DataFrame(dict(year=number//12,month=number%12+1,day=1),index=orig.index))
 return first+pd.to_timedelta(np.minimum(orig.dt.day,first.dt.days_in_month)-1,unit='D')+(orig-orig.dt.normalize())

def safe_logratio(a,b):
 ratio=div(a,b);out=np.full(np.shape(ratio),np.nan);np.log(ratio,out=out,where=np.isfinite(ratio)&(ratio>0));return out

def build():
 started=time.time();(S/'data').mkdir(parents=True,exist_ok=True);(S/'reports').mkdir(exist_ok=True)
 a=pd.concat([pd.read_csv(RAW/'train_applications.csv').drop(columns='target'),pd.read_csv(RAW/'test_applications.csv')],ignore_index=True)
 a['decision_datetime']=pd.to_datetime(a.decision_datetime);assert a.application_id.is_unique and a.customer_id.is_unique
 prof=pd.read_csv(RAW/'customer_profiles.csv',usecols=['customer_id','declared_monthly_income']);a=a.merge(prof,on='customer_id',validate='one_to_one');a.index.name='row'
 a.to_parquet(S/'data/applications.parquet',index=False)
 f=pd.DataFrame(dict(application_id=a.application_id,discovery_aux_row=np.arange(len(a))),index=a.index)
 lookup=a.reset_index()[['row','customer_id','decision_datetime']]
 ec=pd.read_csv(RAW/'ecommerce_transactions.csv',engine='pyarrow');ec['transaction_datetime']=pd.to_datetime(ec.transaction_datetime)
 assert ec.order_item_id.is_unique
 ec=ec.merge(lookup,on='customer_id',how='left',validate='many_to_one');assert ec.row.notna().all()
 nfuture=int((ec.transaction_datetime>=ec.decision_datetime).sum());ec=ec[ec.transaction_datetime<ec.decision_datetime].copy()
 code,ids=pd.factorize(ec.order_id,sort=False);first=np.flatnonzero(~ec.order_id.duplicated().to_numpy());n=len(first)
 # Repeated order fields must agree; item rows are preserved for composition.
 for col in ['row','transaction_datetime','order_total','payment_method','order_status']:
  value=ec[col].to_numpy();assert np.all(value==value[first][code]),col
 item_sum=np.bincount(code,weights=ec.item_total,minlength=n);assert np.all(item_sum==ec.order_total.to_numpy()[first])
 cat=pd.Categorical(ec.product_category,categories=CATS).codes;assert (cat>=0).all()
 mask=np.zeros(n,dtype=np.int16);np.bitwise_or.at(mask,code,np.left_shift(1,cat).astype(np.int16));assert (mask>0).all()
 bits=np.array([int(i).bit_count() for i in range(128)])
 orders=ec.iloc[first][['row','transaction_datetime','order_total','payment_method','order_status']].reset_index(drop=True)
 orders['n_items']=np.bincount(code,minlength=n);orders['quantity']=np.bincount(code,weights=ec.quantity,minlength=n)
 orders['mask']=mask;orders['n_categories']=bits[mask]
 squares=np.bincount(code,weights=ec.item_total.to_numpy(float)**2,minlength=n)
 orders['item_value_hhi']=div(squares,orders.order_total.to_numpy(float)**2)
 orders['age']=(orders.row.map(a.decision_datetime)-orders.transaction_datetime).dt.total_seconds()/86400
 orders['event_month']=orders.transaction_datetime.dt.to_period('M').astype(int)
 orders.to_parquet(S/'data/orders.parquet',index=False)
 complete=orders[orders.order_status=='completed'].copy();total=len(a)
 print('Discovery validated item/order grains',len(ec),len(orders),flush=True)
 def aligned(x,fill=np.nan):
  out=x.reindex(f.index);return out.fillna(fill) if np.isfinite(fill) else out
 stats={}
 for lo,hi,key in [(0,30,'recent'),(30,60,'prior'),(0,90,'90')]:
  q=complete[(complete.age>lo)&(complete.age<=hi)].copy();q['multi']=q.n_categories>1
  stats[key]=q.groupby('row').agg(items=('n_items','mean'),categories=('n_categories','mean'),quantity=('quantity','mean'),hhi=('item_value_hhi','mean'),multi=('multi','mean'),n=('row','size'))
 for name,col in [('items_per_order','items'),('categories_per_order','categories'),('quantity_per_order','quantity'),('item_value_concentration','hhi'),('multicategory_share','multi')]:
  values=aligned(stats['recent'][col]);values[aligned(stats['recent'].n,0)<3]=np.nan
  add(f,'discover_basket_'+name+'_30d',values,'basket_structure','mean('+col+') over completed unique orders in30d; >=3 orders')
 for name,col in [('item_complexity_change','items'),('category_complexity_change','categories'),('quantity_change','quantity')]:
  val=aligned(stats['recent'][col])-aligned(stats['prior'][col]);val[(aligned(stats['recent'].n,0)<3)|(aligned(stats['prior'].n,0)<3)]=np.nan
  add(f,'discover_basket_'+name,val,'basket_structure','mean_'+col+'_0_30d - mean_'+col+'_30_60d; >=3 orders each window')
 hist=complete[(complete.age>30)&(complete.age<=180)].groupby(['row','mask']).size().rename('past_n')
 q=complete[complete.age<=30].merge(hist,on=['row','mask'],how='left');q['novel']=q.past_n.isna()
 prior_n=complete[(complete.age>30)&(complete.age<=180)].groupby('row').size()
 novelty=aligned(q.groupby('row').novel.mean());novelty[aligned(prior_n,0)<5]=np.nan
 add(f,'discover_basket_personal_combo_novelty',novelty,'basket_cooccurrence','share completed30d category masks absent in own30:180d history; >=5 prior orders','Category combinations, not product/SKU combinations; no product ID is supplied')
 counts=complete[complete.age<=90].groupby(['row','mask']).size().rename('n').reset_index();den=counts.groupby('row').n.transform('sum');counts['entropy_piece']=-(counts.n/den)*np.log(counts.n/den)
 ent=aligned(counts.groupby('row').entropy_piece.sum());ent[aligned(stats['90'].n,0)<3]=np.nan
 add(f,'discover_basket_combo_entropy',ent,'basket_cooccurrence','entropy(category-presence-mask distribution over completed90d orders); >=3 orders')
 # Personal unit-price anomalies; earlier same-category items are the reference.
 items=ec[(ec.order_status=='completed')&(ec.quantity>0)&(ec.item_total>0)].copy();items['log_unit']=np.log(items.item_total/items.quantity)
 items['age']=(items.decision_datetime-items.transaction_datetime).dt.total_seconds()/86400
 old=items[(items.age>30)&(items.age<=180)].groupby(['row','product_category']).log_unit.agg(['mean','std','size'])
 recent=items[items.age<=30].merge(old,on=['row','product_category'],how='left')
 recent['z']=(recent.log_unit-recent['mean'])/recent['std'].where((recent['std']>1e-6)&(recent['size']>=5));recent['extreme']=np.where(recent.z.notna(),(abs(recent.z)>2.5).astype(float),np.nan)
 for name,values,formula in [
  ('mean_z',recent.groupby('row').z.mean(),'mean(z) among eligible recent items'),('p90_z',recent.groupby('row').z.quantile(.9),'P90(z) among eligible recent items'),
  ('extreme_share',recent.groupby('row').extreme.mean(),'share |z|>2.5 among eligible recent items'),('observed_items',recent.groupby('row').z.count(),'number eligible recent items')]:
  add(f,'discover_unit_price_'+name,aligned(values),'unitprice',formula+'; z=(log(item_total/quantity)-own_same_category_mean_30:180d)/own_same_category_std_30:180d; >=5 prior items',
   'Category-level comparison cannot hold exact product quality constant; zero quantity/value excluded from unit-price statistics only')
 del ec,items,recent,old
 # Historical application events; full matched follow-up and pretrend support.
 prev=pd.read_csv(RAW/'previous_applications.csv',engine='pyarrow');prev['historical_decision']=pd.to_datetime(prev.pop('decision_datetime'))
 prev=prev.merge(lookup,on='customer_id',validate='many_to_one');prev=prev[(prev.historical_decision<prev.decision_datetime)&(~prev.application_id.isin(a.application_id))].copy()
 earliest=orders.groupby('row').transaction_datetime.min()
 methods=['COD','e_wallet','bank_transfer','debit_card','credit_card']
 for decision in ['approved','rejected']:
  candidates=prev[(prev.decision==decision)&((prev.decision_datetime-prev.historical_decision).dt.total_seconds()/86400>=30)].copy()
  candidates=candidates[candidates.historical_decision-pd.Timedelta(days=60)>=candidates.row.map(earliest)]
  events=candidates.sort_values('historical_decision').groupby('row').tail(1).set_index('row').historical_decision
  q=complete.copy();q['rel']=(q.transaction_datetime-q.row.map(events)).dt.total_seconds()/86400
  parts={}
  for lo,hi,key in [(-60,-30,'older'),(-30,0,'before'),(0,30,'after')]:
   z=q[(q.rel>lo)&(q.rel<=hi)].copy();z['card']=z.payment_method=='credit_card';parts[key]=z.groupby('row').agg(spend=('order_total','sum'),n=('row','size'),card=('card','mean'))
  existed=f.index.isin(events.index);sp={key:aligned(g.spend,0) for key,g in parts.items()};ct={key:aligned(g.n,0) for key,g in parts.items()}
  for suffix,values,formula in [
   ('spend_change',(sp['after']-sp['before'])/a.declared_monthly_income,'(completed_order_spend_after30d-spend_before30d)/declared_monthly_income'),
   ('spend_pretrend_adjusted',(sp['after']-2*sp['before']+sp['older'])/a.declared_monthly_income,'(after30d-2*before30d+older30d spending)/declared_income'),
   ('creditcard_shift',aligned(parts['after'].card)-aligned(parts['before'].card),'credit_card_order_share_after30d - before30d'),
   ('orders_support',ct['after']+ct['before'],'completed order count in matched +/-30d event window')]:
   add(f,'discover_event_'+decision+'_'+suffix,np.where(existed,values,np.nan),'application_events',formula,
    'Latest qualifying historical decision only; all post-event days before selected cutoff; pretrend is descriptive, not a causal control group')
  method_counts=[]
  for lo,hi in [(-30,0),(0,30)]:
   tab=pd.crosstab(q.loc[(q.rel>lo)&(q.rel<=hi),'row'],q.loc[(q.rel>lo)&(q.rel<=hi),'payment_method']).reindex(index=f.index,columns=methods,fill_value=0).to_numpy(float)
   method_counts.append(tab)
  p=(method_counts[0]+.5)/(method_counts[0].sum(1,keepdims=True)+2.5)
  t=(method_counts[1]+.5)/(method_counts[1].sum(1,keepdims=True)+2.5);mid=.5*(p+t)
  js=.5*np.sum(p*np.log(p/mid)+t*np.log(t/mid),axis=1)
  enough=(method_counts[0].sum(1)>=3)&(method_counts[1].sum(1)>=3)&existed
  add(f,'discover_event_'+decision+'_payment_js',np.where(enough,js,np.nan),'application_events','Jensen-Shannon divergence of five payment-method order distributions before/after30d; Jeffreys .5 smoothing; >=3 completed orders each side')
 print('Discovery basket and application events READY',flush=True)
 # Nearest historical due event: assign every order at most once.
 repay=pd.read_csv(RAW/'repayment_history.csv',usecols=['customer_id','loan_id','due_date'],engine='pyarrow');repay['due_date']=pd.to_datetime(repay.due_date)
 loans=prev[prev.loan_id.notna()][['loan_id','customer_id']].drop_duplicates();repay=repay.merge(loans,on=['loan_id','customer_id'],validate='many_to_one')
 repay=repay.merge(lookup,on='customer_id',validate='many_to_one');repay=repay[(repay.due_date<repay.decision_datetime-pd.Timedelta(days=7))&(repay.due_date>=repay.decision_datetime-pd.Timedelta(days=180))]
 due=repay[['row','due_date']].drop_duplicates();due=due[due.due_date-pd.Timedelta(days=14)>=due.row.map(earliest)].sort_values(['row','due_date'])
 previous=due.groupby('row').due_date.diff().dt.total_seconds()/86400;following=-due.groupby('row').due_date.diff(-1).dt.total_seconds()/86400
 due['pre_days']=np.minimum(previous.fillna(14)/2,7);due['post_days']=np.minimum(following.fillna(14)/2,7)
 exposure=due.groupby('row').agg(pre=('pre_days','sum'),post=('post_days','sum'),events=('row','size'))
 q=complete[complete.age<=187].sort_values('transaction_datetime')
 if len(due):
  matched=pd.merge_asof(q,due[['row','due_date']].sort_values('due_date'),left_on='transaction_datetime',right_on='due_date',by='row',direction='nearest',tolerance=pd.Timedelta(days=7))
  matched['rel']=(matched.transaction_datetime-matched.due_date).dt.total_seconds()/86400;matched['card']=matched.payment_method=='credit_card'
  summaries={}
  for key,mask in [('pre',matched.rel<0),('post',matched.rel>=0)]:summaries[key]=matched[mask].groupby('row').agg(spend=('order_total','sum'),card=('card','mean'))
  pre_spend=aligned(summaries['pre'].spend,0);post_spend=aligned(summaries['post'].spend,0)
  add(f,'discover_due_daily_spend_logratio',safe_logratio(div(post_spend,aligned(exposure.post)),div(pre_spend,aligned(exposure.pre))),'due_events','log((post_due_completed_spend/post_exposure_days)/(pre_due_spend/pre_exposure_days)); +/-7d nearest-due windows',
   'Window overlap split at midpoints; each order assigned once. This is scheduled-due alignment, not observed DPD timing.')
  add(f,'discover_due_creditcard_shift',aligned(summaries['post'].card)-aligned(summaries['pre'].card),'due_events','post-due credit-card completed-order share minus pre-due share; +/-7d')
  add(f,'discover_due_pre_spend_per_income',div(div(pre_spend,aligned(exposure.pre))*30,a.declared_monthly_income),'due_events','30*pre_due_daily_spend/declared_monthly_income')
  add(f,'discover_due_event_count',aligned(exposure.events,0),'due_events','eligible historical distinct due-event count; mature7d follow-up and prior14d ecom support')
 else:
  for name in ['daily_spend_logratio','creditcard_shift','pre_spend_per_income','event_count']:add(f,'discover_due_'+name,np.full(total,np.nan),'due_events','No eligible events; missing, not zero')
 # Contractual maturity proxies and previous price transitions.
 orig=prev[(prev.decision=='approved')&prev.loan_id.notna()].copy();orig['maturity']=schedule_end(orig.historical_decision,orig.term_months)
 orig['gap']=(orig.maturity-orig.decision_datetime).dt.total_seconds()/86400
 nearest=orig.assign(abs_gap=abs(orig.gap)).sort_values(['abs_gap','historical_decision']).groupby('row').head(1).set_index('row')
 future=orig[orig.gap>=0];past=orig[orig.gap<0]
 f['discovery_aux_maturity_gap']=aligned(nearest.gap)
 for name,val,formula in [
  ('nearest_signed_gap',aligned(nearest.gap),'signed days from selected decision to nearest estimated historical contractual maturity'),
  ('next_days',aligned(future.groupby('row').gap.min()),'minimum nonnegative days to known contractual maturity'),
  ('since_last_days',-aligned(past.groupby('row').gap.max()),'days since most recent estimated contractual maturity'),
  ('maturing30_count',aligned(orig[orig.gap.between(0,30)].groupby('row').size(),0),'count known historical loan contracts maturing in next30d'),
  ('ended30_count',aligned(orig[orig.gap.between(-30,0,inclusive='left')].groupby('row').size(),0),'count known historical loan contracts ending in previous30d'),
  ('maturing30_burden',div(aligned(orig[orig.gap.between(0,30)].groupby('row').scheduled_monthly_installment.sum(),0),a.declared_monthly_income),'scheduled installment sum of contracts maturing30d / declared_income')]:
  add(f,'discover_maturity_'+name,val,'maturity',formula,'Decision date + tenor is a maturity proxy; early settlement, refinancing and origination lags are not observed')
 last=orig.sort_values('historical_decision').groupby('row').tail(1).set_index('row')
 add(f,'discover_pricing_change_last_loan',a.annual_interest_rate-aligned(last.annual_interest_rate),'pricing_transition','current annual_interest_rate - latest previously approved loan rate')
 add(f,'discover_pricing_upward_transition',np.where(aligned(last.annual_interest_rate).notna(),(a.annual_interest_rate>aligned(last.annual_interest_rate)).astype(float),np.nan),'pricing_transition','1 if current APR exceeds last historically approved APR; missing without prior loan')
 add(f,'discover_pricing_transition_age',div((a.decision_datetime-aligned(last.historical_decision)).dt.total_seconds()/86400,np.full(total,365.25)),'pricing_transition','years since latest historically approved loan decision')
 # Applications near a pre-existing contract's estimated end: no future-origin loans.
 pairs=prev[['row','application_id','historical_decision','decision']].merge(orig[['row','application_id','historical_decision','maturity']].rename(columns={'application_id':'origin_application','historical_decision':'origin_decision'}),on='row',how='inner')
 pairs=pairs[(pairs.application_id!=pairs.origin_application)&(pairs.historical_decision>pairs.origin_decision)]
 near=pairs[abs((pairs.historical_decision-pairs.maturity).dt.total_seconds()/86400)<=30].drop_duplicates(['row','application_id'])
 add(f,'discover_maturity_application_alignment',div(aligned(near.groupby('row').size(),0),aligned(prev.groupby('row').size(),0)),'maturity_inquiry','share historical applications within +/-30d of a loan contract end, where that contract originated before the historical application')
 # Bureau inquiry counts are rolling totals, not individual inquiry timestamps.
 bu=pd.read_csv(RAW/'bureau_observations.csv',engine='pyarrow');bu['report_datetime']=pd.to_datetime(bu.report_datetime);bu['available_datetime']=pd.to_datetime(bu.available_datetime)
 bu=bu.merge(lookup,on='customer_id',validate='many_to_one');bu=bu[(bu.available_datetime<=bu.decision_datetime)&(bu.report_datetime<=bu.available_datetime)]
 bu=bu.sort_values(['row','bureau_provider','report_datetime','available_datetime']);bu['inquiry_change']=bu.groupby(['row','bureau_provider']).inquiries_last_6m.diff()
 latest=bu.groupby(['row','bureau_provider']).tail(1);iq=aligned(latest.groupby('row').inquiries_last_6m.max());delta=aligned(latest.groupby('row').inquiry_change.mean())
 proximity=np.exp(-abs(f.discovery_aux_maturity_gap)/30)
 add(f,'discover_maturity_inquiry_pressure',proximity*np.log1p(iq),'maturity_inquiry','exp(-abs(nearest_maturity_gap)/30)*log1p(max_latest_provider_inquiries6m)',
  'Broad inquiry intensity aligned to estimated maturity; no individual inquiry event times are available')
 add(f,'discover_maturity_inquiry_net_change',proximity*delta,'maturity_inquiry','maturity_proximity * mean_provider(latest_inquiries6m - preceding_same_provider_inquiries6m)',
  'Difference of overlapping rolling counts is a net change, not new inquiry count. Provider counts are not summed.')
 add(f,'discover_maturity_credit_demand',proximity*np.log1p(iq)*div(a.loan_amount,a.declared_monthly_income),'maturity_inquiry','maturity_proximity*log1p(max_inquiries6m)*new_loan_amount/declared_income')
 # Fold-local co-occurrence and underwriting tier definitions.
 for name,family,formula,quarantine in [
  ('discover_basket_combo_surprisal','basket_cooccurrence','mean(-log(P(category_mask))) over90d orders; Dirichlet .5, fold-local references before query month; own customer excluded for correction rows',False),
  ('discover_basket_rare_combo_share','basket_cooccurrence','share90d orders with posterior category-mask probability <.005; >=3 query orders and >=1000 reference orders',False),
  ('discover_basket_pair_lift_surprisal','basket_cooccurrence','mean over basket pairs of -log(P(pair)/(P(category_a)*P(category_b))); fold/time-local reference',False),
  ('discover_pricing_tier_surprise','pricing_tier','-log(P(actual_APR_tier|observed credit characteristics)); cross-fitted on correction rows; APR not default is nuisance target',False),
  ('discover_pricing_tier_expected_gap','pricing_tier','actual_APR - sum(tier_APR*predicted_tier_probability)',False),
  ('discover_pricing_tier_uncertainty','pricing_tier','entropy of predicted five-tier APR distribution',False),
  ('discover_pricing_tier_higher_mass','pricing_tier','predicted probability mass at APR tiers above actual tier',False),
  ('discoverq_pricing_dpd_surprise','pricing_dpd','-log(P(actual_APR_tier|credit characteristics plus current-snapshot historical DPD)); cross-fitted',True),
  ('discoverq_pricing_dpd_expected_gap','pricing_dpd','actual_APR - expected_APR conditioned additionally on DPD; snapshot provenance unverified',True)]:
  CATALOG.append(dict(feature_name=name,family=family,formula=formula,status='QUARANTINED_FOLD_LOCAL' if quarantine else 'FOLD_LOCAL',source='Raw basket masks and audited fold-local financial features',cutoff_rule='References fitted only on correction rows; nuisance3-fold crossfit; basket population capped before query decision month',leakage_risk='Unverified DPD snapshot' if quarantine else 'Incrementally low; inherited baseline uncertainty',caveat='Tier is policy-generated; do not infer causal risk. No default labels in nuisance fitting.',validation_status='Unvalidated',priority='P1'))
 for col in f.columns.difference(['application_id']):f[col]=pd.to_numeric(f[col]).replace([np.inf,-np.inf],np.nan)
 f.to_parquet(S/'data/raw_features.parquet',index=False);pd.DataFrame(CATALOG).to_csv(S/'reports/FEATURE_CATALOG.csv',index=False)
 dump(dict(status='READY',rows=total,direct_columns=len(f.columns)-1,catalog_features=len(CATALOG),items_validated=len(code),orders_validated=n,
  future_events_excluded=nfuture,order_item_totals_match=True,quantity_zero_unit_price_excluded=True,no_default_labels=True,
  elapsed_seconds=time.time()-started,feature_sha256=sha(S/'data/raw_features.parquet'),orders_sha256=sha(S/'data/orders.parquet')),
  S/'reports/FEATURE_STATUS.json');print('Discovery raw features READY',flush=True)

def pricing_matrix(d,dpd=False):
 cols=['declared_monthly_income','loan_amount','aux_account_income','aux_bureau_score','aux_bureau_utilization','aux_account_months','ext_aux_contractual_active_scheduled','age','employment_tenure_months']
 if dpd:cols+=['ext_aux_dpd_mean','ext_aux_recent_dpd15_share','ext_aux_recent_dpd15_n']
 x=d[cols].to_numpy(float);x=np.sign(x)*np.log1p(abs(x))
 for cat,levels in [('employment_type',['salaried','self_employed','variable_income']),('loan_purpose',['education','home_improvement','household','working_capital','medical'])]:
  x=np.column_stack([x,np.array([(d[cat]==v).to_numpy(float) for v in levels]).T])
 return x

def fit_tier(x,y):
 classes=np.argmin(abs(y[:,None]-TIERS),axis=1);assert np.max(abs(y-TIERS[classes]))<1e-8
 prior=(np.bincount(classes,minlength=5)+.5)/(len(classes)+2.5)
 if len(np.unique(classes))<2:return dict(prior=prior,model=None)
 m=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=20,learning_rate=.05,early_stopping=False,random_state=31027)
 with threadpool_limits(limits=2):m.fit(x,classes)
 return dict(prior=prior,model=m)

def predict_tier(bundle,x):
 p=np.tile(bundle['prior'],(len(x),1))
 if bundle['model'] is not None:
  p[:]=0
  with threadpool_limits(limits=2):p[:,bundle['model'].classes_]=bundle['model'].predict_proba(x)
  p=.99*p+.01*bundle['prior']
 assert np.allclose(p.sum(1),1) and (p>0).all();return p

def population_scores(query_ids,query_months,reference_ids,orders,hist90,own_exclude=False):
 """Only reference orders from months strictly earlier than each query month."""
 selected=orders[orders.row.isin(reference_ids)&(orders.order_status=='completed')]
 cats=np.array([[(m>>j)&1 for j in range(7)] for m in range(128)],float)
 pair=np.array([cats[:,i]*cats[:,j] for i,j in itertools.combinations(range(7),2)]).T
 result=np.full((len(query_ids),3),np.nan);refs={}
 own=np.zeros((len(query_ids),128),float)
 if own_exclude:
  lookup=pd.Series(np.arange(len(query_ids)),index=query_ids)
  z=selected.copy();z['querypos']=z.row.map(lookup);month=pd.Series(query_months,index=query_ids)
  z=z[z.event_month<z.row.map(month)]
  np.add.at(own,(z.querypos.to_numpy(int),z['mask'].to_numpy(int)),1)
 for month in np.unique(query_months):
  rows=np.flatnonzero(query_months==month);past=selected[selected.event_month<month]
  counts=np.bincount(past['mask'],minlength=128).astype(float);refs[int(month)]=counts
  c=np.tile(counts,(len(rows),1))-own[rows];assert (c>=0).all();n=c.sum(1)
  posterior=c+.5;posterior[:,0]=0;posterior/=posterior.sum(1,keepdims=True)
  marginal=posterior@cats;joint=posterior@pair
  products=np.column_stack([marginal[:,i]*marginal[:,j] for i,j in itertools.combinations(range(7),2)])
  pair_surprise=-np.log(joint/products)
  lifts=(pair_surprise@pair.T)/np.maximum(pair.sum(1),1)[None,:]
  h=hist90[query_ids[rows]];den=h.sum(1);mask_has_pair=pair.sum(1)>0
  out=np.column_stack([div(np.sum(-np.log(posterior[:,1:])*h[:,1:],axis=1),den),
   div(np.sum((posterior<.005)*h,axis=1),den),div(np.sum(lifts[:,mask_has_pair]*h[:,mask_has_pair],axis=1),h[:,mask_has_pair].sum(1))])
  out[(n<1000)|(den<3)]=np.nan;result[rows]=out
 return result,refs

def fold(fold):
 (S/'data'/f'fold{fold}').mkdir(parents=True,exist_ok=True);raw=pd.read_parquet(S/'data/raw_features.parquet')
 a=pd.read_parquet(S/'data/applications.parquet');tr=pd.read_parquet(ROOT/'data/train_features.parquet');te=pd.read_parquet(ROOT/'data/test_features.parquet')
 idx=np.load(ROOT/'artifacts'/f'fold{fold}'/'indices.npz');hi,vi=idx['correction'],idx['validation'];assert not np.intersect1d(hi,vi).size
 parts={}
 for name,base in [('correction',tr.iloc[hi]),('validation',tr.iloc[vi]),('test',te)]:
  existing=pd.read_parquet(V/'extension/data'/f'fold{fold}'/(name+'.parquet'));d=combined(base,existing)
  parts[name]=d.merge(raw,on='application_id',validate='one_to_one',how='left')
 orders=pd.read_parquet(S/'data/orders.parquet');q=orders[(orders.order_status=='completed')&(orders.age<=90)]
 hist90=np.zeros((len(a),128),np.int32);np.add.at(hist90,(q.row.to_numpy(int),q['mask'].to_numpy(int)),1)
 month=a.decision_datetime.dt.to_period('M').astype(int).to_numpy();ref_ids=parts['correction'].discovery_aux_row.to_numpy(int);reference_cache={}
 for name,d in parts.items():
  ids=d.discovery_aux_row.to_numpy(int);values,refs=population_scores(ids,month[ids],ref_ids,orders,hist90,own_exclude=name=='correction')
  for j,col in enumerate(['discover_basket_combo_surprisal','discover_basket_rare_combo_share','discover_basket_pair_lift_surprisal']):d[col]=values[:,j]
  reference_cache[name]=refs
 print('Discovery fold population references READY',fold,flush=True)
 h=parts['correction'];models={}
 for dpd in [False,True]:
  x=pricing_matrix(h,dpd);y=h.annual_interest_rate.to_numpy(float);p=np.full((len(h),5),np.nan);inner=[]
  for ti,qi in KFold(3,shuffle=True,random_state=31027+fold).split(x):
   assert not np.intersect1d(ti,qi).size
   m=fit_tier(x[ti],y[ti]);p[qi]=predict_tier(m,x[qi]);inner.append(dict(model=m,train_indices=ti,query_indices=qi))
  final=fit_tier(x,y);models['dpd' if dpd else 'base']=dict(final=final,inner=inner)
  for name,d in parts.items():
   probabilities=p if name=='correction' else predict_tier(final,pricing_matrix(d,dpd));actual=d.annual_interest_rate.to_numpy(float)
   tier=np.argmin(abs(actual[:,None]-TIERS),axis=1);actualp=probabilities[np.arange(len(d)),tier]
   if dpd:
    d['discoverq_pricing_dpd_surprise']=-np.log(actualp);d['discoverq_pricing_dpd_expected_gap']=actual-probabilities@TIERS
   else:
    d['discover_pricing_tier_surprise']=-np.log(actualp);d['discover_pricing_tier_expected_gap']=actual-probabilities@TIERS
    d['discover_pricing_tier_uncertainty']=-np.sum(probabilities*np.log(probabilities),axis=1)
    d['discover_pricing_tier_higher_mass']=np.sum(probabilities*(TIERS[None,:]>actual[:,None]),axis=1)
 folder=S/'data'/f'fold{fold}'
 for name,d in parts.items():
  keep=['application_id']+[c for c in d if c.startswith(('v7_','v7q_','aux_','ext_aux_','risk_','riskq_','discover_','discoverq_','discovery_aux_'))]
  d[keep].to_parquet(folder/(name+'.parquet'),index=False)
 joblib.dump(dict(pricing=models,basket_reference_histograms=reference_cache,reference_customer_rows=ref_ids),folder/'nuisance.joblib',compress=3)
 dump(dict(status='PASS',fold=fold,correction_ids_sha256=ids_hash(tr,hi),validation_ids_sha256=ids_hash(tr,vi),
  no_default_target_in_nuisance=True,APR_not_default_is_nuisance_target=True,no_installment_or_APR_in_pricing_predictors=True,
  population_month_strictly_before_query=True,own_customer_excluded_in_correction_reference=True,
  outer_validation_and_test_excluded_from_reference_fit=True,inner_crossfit_splits=3,
  raw_sha256=sha(S/'data/raw_features.parquet'),orders_sha256=sha(S/'data/orders.parquet')),folder/'AUDIT.json')
 print('Discovery fold READY',fold,flush=True)

def checks():
 origin=pd.Series(pd.to_datetime(['2024-01-31 10:00:00','2024-02-29 10:00:00']))
 assert schedule_end(origin,pd.Series([1,12])).tolist()==[pd.Timestamp('2024-02-29 10:00:00'),pd.Timestamp('2025-02-28 10:00:00')]
 assert np.isnan(safe_logratio([0],[2])[0]) and np.isnan(safe_logratio([2],[0])[0])
 rng=np.random.default_rng(31027);x=rng.normal(size=(200,3));y=np.tile(TIERS,40)
 m=fit_tier(x,y);p=predict_tier(m,x);assert p.shape==(200,5) and np.allclose(p.sum(1),1)
 # A future population order must not change an earlier query's reference score.
 orders=pd.DataFrame(dict(row=np.zeros(1100,dtype=int),order_status=['completed']*1100,event_month=np.zeros(1100,dtype=int),mask=np.ones(1100,dtype=int)))
 hist=np.zeros((2,128),int);hist[1,1]=4
 a,_=population_scores(np.array([1]),np.array([1]),np.array([0]),orders,hist)
 extra=pd.concat([orders,pd.DataFrame(dict(row=[0],order_status=['completed'],event_month=[2],mask=[3]))],ignore_index=True)
 b,_=population_scores(np.array([1]),np.array([1]),np.array([0]),extra,hist);assert np.allclose(a,b,equal_nan=True)
 (S/'reports').mkdir(parents=True,exist_ok=True);dump(dict(status='PASS',source_sha256=sha(__file__),checks=['calendar_maturity_end_of_month','undefined_spend_ratios','five_tier_probability_normalization','future_population_event_exclusion']),S/'reports/UNIT_CHECKS.json')
 print('Discovery integrity checks PASS',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int);p.add_argument('--checks',action='store_true');args=p.parse_args()
 if args.checks:checks()
 elif args.fold is not None:fold(args.fold)
 else:build()
