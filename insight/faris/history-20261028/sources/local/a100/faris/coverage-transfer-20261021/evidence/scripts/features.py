"""V7 raw-event features. No default labels enter feature construction.

Run with the pinned AI Center environment. Original baseline assets are read only.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import sys,json,hashlib,time,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy,ks_2samp,chi2_contingency
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from baseline import dump,sha
V=ROOT/'v7'; RAW=ROOT.parents[1]/'data/raw'
DAY=86400e9
CATS=['groceries','fashion','electronics','home','health','education','personal_care']
PAY=['COD','e_wallet','bank_transfer','debit_card','credit_card']
PURPOSE={'medical':['health'],'education':['education'],'home_improvement':['home'],'household':['groceries','home']}
CATALOG=[]

def div(a,b):
 a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
 out=np.full(a.shape,np.nan);np.divide(a,b,out=out,where=np.isfinite(a)&np.isfinite(b)&(b>1e-8));return out.item() if out.ndim==0 else out

def longest(mask):
 best=cur=0
 for yes in mask:
  cur=cur+1 if yes else 0;best=max(cur,best)
 return best

def add(f,name,values,family,formula,source,rule='All events strictly before selected decision; complete account months only',caveat=''):
 key='v7_'+name;f[key]=values
 CATALOG.append(dict(feature_name=key,family=family,formula=formula,source=source,cutoff_rule=rule,
  status='IMPLEMENTED',leakage_risk='Low (incremental; baseline has inherited unverified features)',caveat=caveat,
  validation_status='Unvalidated',priority='P0' if family in ['stress','dna','resilience'] else 'P1'))

def audit_dpd(apps):
 r=pd.read_csv(RAW/'repayment_history.csv',engine='pyarrow');p=pd.read_csv(RAW/'previous_applications.csv',engine='pyarrow')
 r['due_date']=pd.to_datetime(r.due_date);r['last_payment_datetime']=pd.to_datetime(r.last_payment_datetime)
 r=r.merge(apps[['customer_id','decision_datetime']],on='customer_id',how='left',validate='many_to_one')
 p['decision_datetime']=pd.to_datetime(p.decision_datetime)
 loans=p[p.loan_id.notna()][['loan_id','customer_id','decision_datetime']].rename(columns={'decision_datetime':'loan_decision'})
 assert not loans.loan_id.duplicated().any()
 r=r.merge(loans,on=['loan_id','customer_id'],how='left',validate='many_to_one')
 due=(r.decision_datetime.dt.normalize()-r.due_date.dt.normalize()).dt.days.clip(lower=0)
 allocation=(r.last_payment_datetime.dt.normalize()-r.due_date.dt.normalize()).dt.days.clip(lower=0)
 unpaid=r.amount_paid<r.amount_due
 expected=np.where(unpaid,due,0)
 maxlower=np.maximum(r.days_past_due,allocation.fillna(0))
 out=dict(rows=len(r),unique_installments=int(r.installment_id.nunique()),
  orphan_loans=int(r.loan_decision.isna().sum()),loan_after_cutoff=int((r.loan_decision>=r.decision_datetime).sum()),
  due_after_cutoff=int((r.due_date>r.decision_datetime).sum()),last_allocation_at_or_after_cutoff=int((r.last_payment_datetime>=r.decision_datetime).sum()),
  current_dpd_matches_unpaid_calendar_age=int((r.days_past_due==expected).sum()),
  current_dpd_formula_mismatches=int((r.days_past_due!=expected).sum()),
  max_dpd_less_than_current=int((r.max_days_past_due<r.days_past_due).sum()),
  max_dpd_above_age_at_cutoff=int((r.max_days_past_due>due).sum()),
  max_dpd_below_last_allocation_delay_or_current=int((r.max_days_past_due<maxlower).sum()),
  amount_paid_above_due=int((r.amount_paid>r.amount_due).sum()),
  null_allocation_positive_paid=int((r.last_payment_datetime.isna()&(r.amount_paid>0)).sum()),
  current_dpd_quantiles=r.days_past_due.quantile([0,.5,.9,.99,1]).to_dict(),
  max_dpd_quantiles=r.max_days_past_due.quantile([0,.5,.9,.99,1]).to_dict(),
  semantics='Dictionary explicitly states last allocation strictly before cutoff; it does not explicitly date DPD/maxDPD/amount_paid snapshots. Coherence is evidence, not proof of provenance.',
  certification='UNVERIFIED_UNTIL_SOURCE_SNAPSHOT_PROVENANCE',
  trusted_new_features_use='Allocation timestamps and scheduled amount_due only; no DPD or amount_paid derivatives')
 dump(out,V/'reports/DPD_SEMANTICS_AUDIT.json');return out

def build():
 started=time.time();(V/'data').mkdir(parents=True,exist_ok=True);(V/'reports').mkdir(exist_ok=True)
 train=pd.read_csv(RAW/'train_applications.csv',engine='pyarrow');test=pd.read_csv(RAW/'test_applications.csv',engine='pyarrow')
 apps=pd.concat([train.drop(columns='target'),test],ignore_index=True);apps['decision_datetime']=pd.to_datetime(apps.decision_datetime)
 assert apps.application_id.is_unique and apps.customer_id.is_unique
 profile=pd.read_csv(RAW/'customer_profiles.csv',engine='pyarrow');assert profile.customer_id.is_unique
 apps=apps.merge(profile,on='customer_id',validate='one_to_one');apps.index.name='row'
 assert len(apps)==100000
 f=pd.DataFrame(index=apps.index);f['application_id']=apps.application_id
 lookup=apps.reset_index()[['row','customer_id','decision_datetime']]
 audit={'dictionary_sha256':sha(RAW/'DATA_DICTIONARY.md'),'rows':len(apps),'unique_application_per_customer':True,
  'train_test_customer_overlap':len(set(train.customer_id)&set(test.customer_id)),'tables':{},'checks':{}}
 def read(name,timestamps):
  d=pd.read_csv(RAW/(name+'.csv'),engine='pyarrow')
  for c in timestamps:d[c]=pd.to_datetime(d[c])
  audit['tables'][name]={'rows_raw':len(d),'columns':d.columns.tolist(),
   'date_ranges':{c:[None if pd.isna(d[c].min()) else str(d[c].min()),None if pd.isna(d[c].max()) else str(d[c].max())] for c in timestamps}}
  d=d.merge(lookup,on='customer_id',how='left',validate='many_to_one');assert d.row.notna().all();return d
 def place(series,default=np.nan):return series.reindex(f.index).fillna(default) if not pd.isna(default) else series.reindex(f.index)
 # Account: only complete calendar months, retain explicit denominator.
 ac=read('account_behavior',['month']);assert not ac.duplicated(['customer_id','month']).any()
 eligible=ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime
 audit['tables']['account_behavior']['excluded_incomplete_or_future']=int((~eligible).sum());ac=ac[eligible].copy()
 assert (ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime).all()
 ac['calendar_days']=ac.month.dt.days_in_month;assert (ac.low_balance_days.between(0,ac.calendar_days)).all()
 ac['net']=ac.inflows-ac.outflows
 ac=ac.sort_values(['row','month']);recent=ac[ac.month>=ac.decision_datetime.dt.to_period('M').dt.to_timestamp()-pd.DateOffset(months=3)]
 ag=recent.groupby('row').agg(income=('income_like_credits','mean'),inflow=('inflows','mean'),outflow=('outflows','mean'),
  low=('low_balance_days','sum'),days=('calendar_days','sum'),min_balance=('minimum_balance','min'),net=('net','mean'),months=('month','size'))
 latest=ac.groupby('row').last();income=place(ag.income);outflow=place(ag.outflow);balance=place(latest.closing_balance)
 installment=apps.proposed_monthly_installment;declared=apps.declared_monthly_income
 for name,val in {'account_income':income,'account_outflow':outflow,'account_balance':balance,'account_low_share':div(place(ag.low),place(ag.days)),
  'account_months':place(ag.months,0),'account_net':place(ag.net),'declared_income':declared,'installment':installment,'loan_amount':apps.loan_amount}.items():f['aux_'+name]=np.asarray(val)
 add(f,'resilience_income_shock_10',div(.9*income-outflow,installment),'resilience','(.9*mean_income_like_credits_3m - mean_outflows_3m)/new_installment','account_behavior,train/test_applications')
 add(f,'resilience_income_shock_20',div(.8*income-outflow,installment),'resilience','(.8*mean_income_like_credits_3m - mean_outflows_3m)/new_installment','account_behavior,train/test_applications')
 add(f,'resilience_break_even_shock',div(income-outflow-installment,income),'resilience','(income_3m-outflows_3m-new_installment)/income_3m; negative values retained','account_behavior,train/test_applications')
 add(f,'resilience_250k_buffer_rate',div(place(ag.low),place(ag.days)),'resilience','sum(low_balance_days_3m)/sum(days_in_observed_calendar_months_3m)','account_behavior',caveat='Observed months only; absent months are not assumed complete')
 run=ac.groupby('row').apply(lambda g:longest((g.net<0).to_numpy()) if np.all(np.diff(g.month.dt.to_period('M').astype(int))==1) else np.nan,include_groups=False)
 add(f,'resilience_cashflow_run_length',place(run),'resilience','Longest consecutive negative-flow run; NA if calendar gaps exist','account_behavior')
 add(f,'resilience_loan_vs_min_balance',div(installment,place(ag.min_balance)),'resilience','new_installment/min(minimum_balance_3m); nonpositive denominator undefined','account_behavior,train/test_applications')
 add(f,'resilience_reported_income_gap',div(declared-income,declared),'resilience','(declared_income-mean_observed_income_like_credits_3m)/declared_income','customer_profiles,account_behavior')
 add(f,'resilience_reserve_months_stressed',div(balance,np.maximum(outflow+installment-.8*income,0)),'resilience','latest_complete_month_closing_balance / max(outflow+new_installment-.8*income,0); zero deficit -> NA','account_behavior,train/test_applications',caveat='Reserve is partial account visibility, not total household assets')
 f['aux_stressed_no_deficit']=np.where(income.notna(),(outflow+installment-.8*income<=0).astype(float),np.nan)
 # E-commerce: validate repeated order fields before reducing the grain.
 ec=read('ecommerce_transactions',['transaction_datetime']);assert ec.order_item_id.is_unique
 order_fields=['customer_id','transaction_datetime','order_total','payment_method','order_status']
 consistency=ec.groupby('order_id',sort=False)[order_fields].nunique(dropna=False)
 assert (consistency<=1).all().all()
 totals=ec.groupby('order_id',sort=False).agg(item_sum=('item_total','sum'),order_total=('order_total','first'))
 assert (totals.item_sum==totals.order_total).all()
 audit['checks']['ecommerce_order_consistency']=True;audit['checks']['unique_item_ids']=True
 audit['tables']['ecommerce_transactions']['vocabulary']={c:sorted(ec[c].unique().tolist()) for c in ['product_category','payment_method','order_status']}
 assert set(ec.product_category)<=set(CATS) and set(ec.payment_method)<=set(PAY)
 allowed=ec.transaction_datetime<ec.decision_datetime;audit['tables']['ecommerce_transactions']['excluded_post_cutoff']=int((~allowed).sum());ec=ec[allowed].copy()
 ec['age']=(ec.decision_datetime-ec.transaction_datetime).dt.total_seconds()/86400
 orders=ec.drop_duplicates('order_id').copy();orders=orders.sort_values(['row','transaction_datetime','order_id'])
 completed=orders[orders.order_status=='completed'].copy();items=ec[ec.order_status=='completed'].copy()
 assert np.all(items.age>0)
 sums={};counts={}
 for lo,hi,key in [(0,30,'30'),(30,60,'prior30'),(0,90,'90')]:
  e=completed[completed.age.between(lo,hi,inclusive='right')];sums[key]=place(e.groupby('row').order_total.sum(),0);counts[key]=place(e.groupby('row').size(),0)
 p95=place(completed[completed.age<=90].groupby('row').order_total.quantile(.95))
 f['aux_spend30']=sums['30'];f['aux_spend_prior30']=sums['prior30'];f['aux_spend90']=sums['90'];f['aux_orders90']=counts['90']
 f['aux_ecommerce_observed_days']=place(orders.groupby('row').age.max())
 acceleration=div(sums['30'],sums['prior30'])
 add(f,'stress_spend_cash_gap',div(sums['30'],income),'stress','completed_unique_order_spend_30d/mean_income_like_credits_complete_3m','ecommerce_transactions,account_behavior')
 add(f,'stress_burst_lowbalance',acceleration*f.aux_account_low_share,'stress','(spend_0_30d/spend_30_60d)*low_balance_calendar_day_share_3m; zero prior spend undefined','ecommerce_transactions,account_behavior')
 add(f,'stress_tail_purchase_buffer',div(p95,balance),'stress','P95(completed_order_total_90d)/latest_complete_month_closing_balance','ecommerce_transactions,account_behavior')
 card=[]
 for lo,hi in [(0,30),(30,60)]:
  e=completed[(completed.age>lo)&(completed.age<=hi)];card.append(place(e.groupby('row').payment_method.apply(lambda x:(x=='credit_card').mean())))
 f['aux_card_shift']=card[0]-card[1]
 # Order dominant category from item spending, deterministic ties.
 dominant=items.groupby(['order_id','product_category'],sort=False).item_total.sum().reset_index().sort_values(['order_id','item_total','product_category'],ascending=[True,False,True]).drop_duplicates('order_id').set_index('order_id').product_category
 completed['dominant_category']=completed.order_id.map(dominant)
 dna_names=['dna_order_value_gini','dna_top10_spending_share','dna_microburst_ratio','dna_interarrival_irregularity','dna_category_switch_rate','dna_category_payment_mutual_info','dna_large_basket_cancel_rate','dna_spending_phase_entropy','dna_order_clustering_index']
 dna=np.full((len(f),len(dna_names)),np.nan);regimes=np.full((len(f),4),np.nan)
 ordergroups={i:g for i,g in orders[orders.age<=90].groupby('row',sort=False)}
 for counter,(i,g) in enumerate(completed[completed.age<=90].groupby('row',sort=False)):
  vals=g.order_total.to_numpy(float);n=len(vals);vals_sorted=np.sort(vals);total=vals.sum()
  t=g.transaction_datetime.astype('int64').to_numpy()/DAY;dt=np.diff(t)
  cats=pd.Categorical(g.dominant_category,categories=CATS).codes;pay=pd.Categorical(g.payment_method,categories=PAY).codes
  if n>=3 and total>0:
   dna[i,0]=2*np.dot(np.arange(1,n+1),vals_sorted)/(n*total)-(n+1)/n
   dna[i,1]=vals_sorted[-max(1,math.ceil(.1*n)):].sum()/total
  if n>=3:
   dna[i,2]=np.mean(dt<1);dna[i,3]=div(dt.std(ddof=1),dt.mean());dna[i,4]=np.mean(cats[1:]!=cats[:-1])
   hist=np.bincount((cats*len(PAY)+pay),minlength=len(CATS)*len(PAY)).reshape(len(CATS),len(PAY))/n
   independent=hist.sum(1,keepdims=True)*hist.sum(0,keepdims=True);nz=hist>0
   # Miller-Madow first-order finite sample correction, floored at zero.
   mi=np.sum(hist[nz]*np.log(hist[nz]/independent[nz]));kc=(hist.sum(1)>0).sum();kp=(hist.sum(0)>0).sum()
   dna[i,5]=max(0,mi-(kc-1)*(kp-1)/(2*n)) if n>=20 else np.nan
   allg=ordergroups[i];large=allg.order_total>=np.quantile(vals,.9)
   dna[i,6]=(allg.loc[large,'order_status']=='cancelled').mean() if large.sum()>=3 else np.nan
   phases=g.transaction_datetime.dt.dayofweek.to_numpy();dna[i,7]=entropy(np.bincount(phases,minlength=7))
   if n>=10 and f.loc[i,'aux_ecommerce_observed_days']>=90:
    daily=np.bincount(np.floor(g.age.to_numpy()).clip(0,89).astype(int),minlength=90)
    dna[i,8]=div(daily.var(ddof=1),daily.mean())
  recent=g[g.age<=30];prior=g[(g.age>30)&(g.age<=60)]
  if len(recent)>=10 and len(prior)>=10:
   regimes[i,0]=ks_2samp(recent.order_total,prior.order_total).pvalue
   for j,col,levels in [(1,'dominant_category',CATS),(2,'payment_method',PAY)]:
    table=np.array([[sum(part[col]==lev) for lev in levels] for part in [recent,prior]])
    table=table[:,table.sum(0)>0]
    if table.shape[1]>1:
     _,pv,_,expected=chi2_contingency(table,correction=False)
     if (expected>=5).all():regimes[i,j]=pv
  if counter%25000==0:print('DNA customers',counter,flush=True)
 formulas=['2*sum(rank*sorted_order_total)/(n*sum_order_total)-(n+1)/n; n>=3',
  'sum(largest ceil(.1*n) completed order values)/sum(values); n>=3','mean(interarrival_days<1); n>=3',
  'std(interarrival_days,ddof=1)/mean(interarrival_days); n>=3','mean(dominant_category_i!=dominant_category_i-1); n>=3',
  'max(0,empirical_MI(category,payment)-(nonempty_categories-1)*(nonempty_methods-1)/(2*n)); n>=20',
  'cancelled/all orders above personal completed-order P90; >=3 eligible large orders','entropy(7 weekday completed-order counts); n>=3',
  'variance(90 daily order counts,ddof=1)/mean(counts); >=10 orders and >=90d observed history']
 for j,name in enumerate(dna_names):add(f,name,dna[:,j],'dna',formulas[j],'ecommerce_transactions',caveat='90d window; finite sample support and missing indicators retained. Clustering assumes recorded exposure, not verified platform enrollment.')
 essential=place(items[(items.age<=90)&items.product_category.isin(['groceries','health'])].groupby('row').item_total.sum(),0)
 add(f,'dna_tail_to_groceries_health',div(p95,essential),'dna','P95(order_total_90d)/item_spend_90d_in_groceries_and_health','ecommerce_transactions',caveat='Explicit exploratory category grouping; not a claim all health/grocery spending is essential')
 # Category consistency uses item values, never repeated order_total.
 mapped=apps.loan_purpose.isin(PURPOSE)
 for lo,hi,key in [(0,30,'recent'),(30,60,'prior'),(0,90,'all')]:
  e=items[(items.age>lo)&(items.age<=hi)].copy();e['purpose']=e.row.map(apps.loan_purpose)
  aligned=np.array([cat in PURPOSE.get(purpose,[]) for cat,purpose in zip(e.product_category,e.purpose)])
  aligned_sum=place(e[aligned].groupby('row').item_total.sum(),0);full=place(e.groupby('row').item_total.sum(),0)
  f['aux_purpose_'+key]=np.where(mapped,div(aligned_sum,full),np.nan)
 add(f,'purpose_category_alignment',f.aux_purpose_all,'purpose','aligned_category_item_spend_90d/all_completed_item_spend_90d','ecommerce_transactions,train/test_applications',caveat='Exploratory mapping: medical=health; education=education; home_improvement=home; household=groceries+home; working_capital=undefined')
 add(f,'purpose_spending_shift',f.aux_purpose_recent-f.aux_purpose_prior,'purpose','aligned spending share_0_30d - aligned spending share_30_60d','ecommerce_transactions,train/test_applications')
 # Monthly account/ecommerce agreement on exactly matched complete months.
 items['month']=items.transaction_datetime.dt.to_period('M').dt.to_timestamp()
 em=items.groupby(['row','month']).item_total.sum().rename('ecom').reset_index()
 aligned_months=ac[['row','month','outflows']].merge(em,on=['row','month'],how='left').fillna({'ecom':0})
 am=aligned_months.groupby('row').agg(ecom=('ecom','sum'),outflow=('outflows','sum'))
 add(f,'contradiction_bank_ecom_share',div(place(am.ecom),place(am.outflow)),'contradiction','sum(completed item spend in observed complete account months)/sum(outflows in those same months)','ecommerce_transactions,account_behavior',caveat='External visibility differs; ratio >1 can be valid')
 add(f,'contradiction_income_cashflow',np.log1p(declared)-np.log1p(income),'contradiction','log1p(declared_income)-log1p(mean_income_like_credits_3m)','customer_profiles,account_behavior')
 # Previous applications and event-aligned response after latest mature rejection.
 prev=read('previous_applications',['application_datetime','decision_datetime']) if False else pd.read_csv(RAW/'previous_applications.csv',engine='pyarrow')
 prev['historical_decision']=pd.to_datetime(prev.pop('decision_datetime'));prev['application_datetime']=pd.to_datetime(prev.application_datetime)
 audit['tables']['previous_applications']={'rows_raw':len(prev),'columns':prev.columns.tolist(),'historical_decision_range':[str(prev.historical_decision.min()),str(prev.historical_decision.max())]}
 prev=prev.merge(lookup,on='customer_id',validate='many_to_one');prev=prev[(prev.historical_decision<prev.decision_datetime)&(~prev.application_id.isin(apps.application_id))].copy()
 assert (prev.historical_decision<prev.decision_datetime).all();assert not prev.application_id.duplicated().any()
 prev['age']=(prev.decision_datetime-prev.historical_decision).dt.total_seconds()/86400
 rej=prev[prev.decision=='rejected'];recentrej=place(rej[rej.age<=30].groupby('row').size(),0);priorrej=place(rej[(rej.age>30)&(rej.age<=180)].groupby('row').size(),0)
 f['aux_rejection_count30']=recentrej
 add(f,'credit_rejection_acceleration',recentrej-priorrej/5,'debt_safe','count(rejected_0_30d)-count(rejected_30_180d)/5','previous_applications',caveat='Difference of monthly rates; previous-application observation start not documented')
 pa=prev.sort_values(['row','historical_decision']).groupby('row').requested_amount.agg(lambda x:np.log(x.iloc[-1]/x.iloc[-2]) if len(x)>=2 and x.iloc[-2]>0 and x.iloc[-1]>0 else np.nan)
 add(f,'credit_amount_ratchet',place(pa),'debt_safe','log(last_requested_amount/penultimate_requested_amount); >=2 historical requests','previous_applications')
 mature=rej[rej.age>=30].sort_values('historical_decision').groupby('row').tail(1)
 cg={i:g for i,g in completed.groupby('row',sort=False)};shift=np.full(len(f),np.nan);rej_count=np.zeros(len(f));pretrend=np.full(len(f),np.nan)
 for i,event in mature.set_index('row').iterrows():
  if i not in cg:continue
  g=cg[i];rel=(g.transaction_datetime-event.historical_decision).dt.total_seconds().to_numpy()/86400
  # At least 60 days pre-event evidence; after window fully matured before selected decision.
  if (event.decision_datetime-event.historical_decision).total_seconds()/86400<30 or rel.min()>-60:continue
  before=np.sum((rel>=-30)&(rel<0));after=np.sum((rel>0)&(rel<=30));older=np.sum((rel>=-60)&(rel<-30))
  # Log rate ratio with Jeffreys .5 event pseudocount; equal 30d windows.
  shift[i]=np.log((after+.5)/(before+.5));pretrend[i]=np.log((before+.5)/(older+.5));rej_count[i]=before+after
 add(f,'stress_rejection_spending_shift',shift,'stress','log((orders_0_30d_after_latest_mature_rejection+.5)/(orders_30d_before+.5)); >=60d prior observed support','previous_applications,ecommerce_transactions',caveat='Association, not causal rejection effect; provider enrollment unknown; equal windows and pretrend control included')
 add(f,'stress_rejection_pretrend_adjusted',shift-pretrend,'stress','post_vs_pre_log_order_rate - pre_vs_older_log_order_rate','previous_applications,ecommerce_transactions')
 f['aux_rejection_response_orders']=rej_count
 # Repayment snapshot investigation is independent of feature-label analysis.
 audit['dpd']=audit_dpd(apps)
 r=pd.read_csv(RAW/'repayment_history.csv',usecols=['loan_id','installment_id','customer_id','due_date','amount_due','last_payment_datetime'],engine='pyarrow')
 r['due_date']=pd.to_datetime(r.due_date);r['last_payment_datetime']=pd.to_datetime(r.last_payment_datetime)
 r=r.merge(lookup,on='customer_id',validate='many_to_one')
 loans=prev[prev.loan_id.notna()][['loan_id','customer_id']].drop_duplicates();r=r.merge(loans,on=['loan_id','customer_id'],validate='many_to_one')
 future=r[r.due_date>r.decision_datetime].groupby('row').amount_due.sum();f['aux_future_scheduled']=place(future,0)
 r=r[(r.due_date<r.decision_datetime)&r.last_payment_datetime.notna()&(r.last_payment_datetime<r.decision_datetime)].copy()
 r['allocation_delay']=(r.last_payment_datetime-r.due_date).dt.total_seconds()/86400;r['delay_positive']=r.allocation_delay.clip(lower=0)
 rates=r.assign(late=(r.allocation_delay>0).astype(float)).groupby('row').agg(late=('late','mean'),n=('late','size'))
 f['aux_allocation_late_share']=place(rates.late);f['aux_allocation_count']=place(rates.n,0)
 threshold=[15,30,45,60,75,90]
 curves=pd.DataFrame({t:r.assign(event=(r.delay_positive>=t).astype(float)).groupby('row').event.mean() for t in threshold}).reindex(f.index)
 add(f,'repay_allocation_severity_auc',np.trapezoid(curves.to_numpy(),x=threshold,axis=1)/75,'debt_safe','trapezoid(P(last_allocation_delay>=threshold),threshold=[15,30,45,60,75,90])/75','repayment_history',caveat='Allocation delay is not DPD, cure, or full settlement time; conditional on timestamp observed')
 bands=pd.cut(r.delay_positive,[-np.inf,0,15,30,45,60,75,90,np.inf],right=False,labels=False)
 tab=pd.crosstab(r.row,bands).reindex(f.index)
 ent=tab.apply(lambda x:entropy(x.dropna().to_numpy()) if x.notna().any() else np.nan,axis=1)
 add(f,'repay_allocation_band_entropy',ent,'debt_safe','entropy(nonoverlapping last-allocation-delay bands [<0,0:15,15:30,...,>=90])','repayment_history',caveat='Not DPD bands; no-payment installments excluded, support supplied')
 r=r.sort_values(['row','loan_id','due_date']);streak=r.groupby(['row','loan_id']).allocation_delay.agg(lambda s:longest((s>0).to_numpy())).groupby('row').max()
 add(f,'repay_allocation_max_late_streak',place(streak),'debt_safe','max within-loan consecutive eligible due installments with last allocation after due date','repayment_history',caveat='Missing last-allocation installment breaks observation continuity; consecutive eligible observations only')
 weighted=r.assign(numerator=r.delay_positive*r.amount_due).groupby('row').agg(num=('numerator','sum'),den=('amount_due','sum'))
 add(f,'repay_allocation_amount_weighted_delay',div(place(weighted.num),place(weighted.den)),'debt_safe','sum(max(last_allocation-due,0)*amount_due)/sum(amount_due)','repayment_history')
 late_loans=r.assign(severe=r.allocation_delay>=30).groupby(['row','loan_id']).severe.max().groupby('row').mean()
 add(f,'repay_allocation_bad_loan_concentration',place(late_loans),'debt_safe','share historical loans with observed last-allocation delay >=30d','repayment_history',caveat='Not a default or DPD30 flag')
 def trajectory(g):
  if len(g)<3:return pd.Series([np.nan,np.nan,np.nan])
  t=(g.due_date-g.due_date.min()).dt.total_seconds().to_numpy()/86400;y=g.delay_positive.to_numpy()
  if np.ptp(t)==0:return pd.Series([np.nan,np.nan,np.nan])
  tc=t-t.mean();slope=np.dot(tc,y-y.mean())/np.dot(tc,tc);res=y-y.mean()-slope*tc
  se=np.sqrt(np.dot(res,res)/max(len(y)-2,1)/np.dot(tc,tc))
  return pd.Series([max(-slope,0),max(slope,0),se])
 tr=r.groupby('row').apply(trajectory,include_groups=False).reindex(f.index)
 for j,name in enumerate(['recovery_strength','deterioration_strength','trajectory_uncertainty']):add(f,name,tr[j],'composites','OLS per-calendar-day last-allocation-delay slope; recovery=max(-slope,0); deterioration=max(slope,0); uncertainty=slope SE; n>=3','repayment_history',caveat='Allocation trajectory only; different historical loans may overlap')
 # Bureau: most recent available observation per provider; no sum across providers.
 bu=read('bureau_observations',['report_datetime','available_datetime']);assert bu.bureau_observation_id.is_unique
 eligible=(bu.available_datetime<=bu.decision_datetime)&(bu.report_datetime<=bu.available_datetime)
 audit['tables']['bureau_observations']['excluded_unavailable_or_invalid']=int((~eligible).sum());bu=bu[eligible].copy()
 assert (bu.available_datetime<=bu.decision_datetime).all()
 latest_bu=bu.sort_values(['report_datetime','available_datetime','bureau_observation_id']).groupby(['row','bureau_provider']).tail(1)
 b=latest_bu.groupby('row').agg(score=('score','mean'),utilization=('utilization','max'),debt=('outstanding_balance','max'))
 f['aux_bureau_score']=place(b.score);f['aux_bureau_utilization']=place(b.utilization);f['aux_bureau_debt_max']=place(b.debt)
 age=(latest_bu.decision_datetime-latest_bu.report_datetime).dt.total_seconds()/86400
 f['aux_bureau_freshness']=place(age.groupby(latest_bu.row).mean())
 bp=latest_bu.pivot(index='row',columns='bureau_provider',values='score').reindex(f.index)
 for provider in ['Bureau_A','Bureau_B','Bureau_C']:f['aux_'+provider]=bp.get(provider,np.nan)
 add(f,'stress_creditcard_shift',f.aux_card_shift*f.aux_bureau_utilization,'stress','(credit_card_order_share_30d-credit_card_order_share_prior30d)*max(latest_provider_utilization)','ecommerce_transactions,bureau_observations')
 add(f,'contradiction_goodscore_badallocation',div(f.aux_bureau_score-300,550)*f.aux_allocation_late_share,'contradiction','((mean_latest_provider_score-300)/550)*share(last_allocation_after_due)','bureau_observations,repayment_history',caveat='Linear within documented common score range, not a provider percentile')
 add(f,'contradiction_schedule_reported_gap',np.log1p(f.aux_future_scheduled)-np.log1p(f.aux_bureau_debt_max),'contradiction','log1p(historical contracted future installment sum)-log1p(max_latest_provider_outstanding)','previous_applications,repayment_history,bureau_observations',caveat='Future scheduled payments include interest; external covered principal differs; not equivalent debt measurements')
 # Regime account test: exact randomization of 3 recent versus 3 earlier months, >=6 observed contiguous months.
 import itertools
 for i,g in ac.groupby('row',sort=False):
  if len(g)<6:continue
  g=g.tail(6)
  if not np.all(np.diff(g.month.dt.to_period('M').astype(int))==1):continue
  vals=g.net.to_numpy(float);observed=abs(vals[:3].mean()-vals[3:].mean())
  stats=[abs(vals[list(c)].mean()-np.delete(vals,list(c)).mean()) for c in itertools.combinations(range(6),3)]
  regimes[i,3]=np.mean(np.array(stats)>=observed-1e-8)
 f['aux_regime_tested_domains']=np.isfinite(regimes).sum(1)
 # Four-domain Bonferroni control; count is undefined if no domain can be tested.
 instab=np.where(np.isfinite(regimes).any(1),np.sum(regimes<=.05/4,axis=1),np.nan)
 add(f,'behavioral_regime_instability',instab,'composites','number of supported order-value/category/payment/cashflow changes with p<=.05/4; NA if no tests supported','ecommerce_transactions,account_behavior',caveat='Conservative, low power; four-domain multiple-test control; no target labels. Six-month exact cashflow test cannot attain .0125 with 3+3 months.')
 # Remaining requested features need training-fold reference models or unsafe snapshots.
 for name,family,formula in [
  ('stress_spending_income_anomaly','stress','(log1p(spend30)-conditional_log_spend_Q50)*new_installment/observed_income'),
  ('contradiction_score_allocation','contradiction','mean_latest_provider_score - crossfit_prediction(score|allocation_history,utilization,report_age)'),
  ('contradiction_bureau_provider_rank','contradiction','std(provider-specific training-reference empirical percentiles), >=2 scored providers'),
  ('purpose_amount_surprise','purpose','log1p(loan_amount)-crossfit_prediction(log_amount|income,tenor,purpose)'),
  ('financial_fragility_index','composites','equal mean of training-median/IQR scaled downside cashflow,low-balance share,installment burden,allocation-late share; >=2 observed')]:
  CATALOG.append(dict(feature_name='v7_'+name,family=family,formula=formula,source='fold-local nuisance transforms of audited raw auxiliary features',cutoff_rule='All preprocessing and nuisance fitting on correction-training rows only; 3-fold crossfit for those training rows',status='FOLD_LOCAL',leakage_risk='Low incrementally; depends on inherited baseline safety',caveat='No default labels in nuisance model; outer validation and test excluded from fitting',validation_status='Unvalidated',priority='P1'))
 blocked=['stress_dpd_cash_coincidence','repay_dpd_severity_auc','repay_dpd_band_entropy','repay_max_late_streak','repay_cure_intensity','repay_bad_loan_concentration','repay_amount_weighted_severity','contradiction_score_dpd','contradiction_goodscore_badpayment','contradiction_credit_exposure']
 for name in blocked:CATALOG.append(dict(feature_name='v7_'+name,family='requested_unverified',formula='See user V7 specification',source='repayment_history and cross-source inputs',cutoff_rule='Requires verified DPD/amount_paid snapshot provenance or equivalent exposure definition',status='BLOCKED_OR_REPLACED_EXPLICITLY',leakage_risk='High until source provenance established',caveat='Safe last-allocation analogues have different names and are not DPD or payment coverage',validation_status='Unvalidated',priority='P2'))
 for col in f.columns.difference(['application_id']):f[col]=pd.to_numeric(f[col],errors='raise').replace([np.inf,-np.inf],np.nan)
 assert f.application_id.is_unique and len(f)==100000
 f.to_parquet(V/'data/features_raw.parquet',index=False)
 pd.DataFrame(CATALOG).to_csv(V/'reports/FEATURE_CATALOG.csv',index=False)
 audit['features']=len([c for c in f if c.startswith('v7_')]);audit['auxiliary_columns']=len([c for c in f if c.startswith('aux_')]);audit['elapsed_seconds']=time.time()-started
 audit['feature_sha256']=sha(V/'data/features_raw.parquet');audit['checks'].update(complete_account_months=True,available_bureau_only=True,historical_applications_only=True,scheduled_historical_loans_only=True,
  no_default_labels_in_features=True,no_unsafe_DPD_columns_in_new_features=True,cancelled_spend_excluded=True)
 audit['trusted_entire_pipeline']=False;audit['inherited_warning']='Locked Kumo/spline/correction already contain unverified DPD/amount_paid and global reference features. New filters do not certify the inherited baseline.'
 dump(audit,V/'reports/TEMPORAL_AUDIT.json');dump(dict(status='READY',rows=len(f),feature_sha256=audit['feature_sha256'],elapsed_seconds=audit['elapsed_seconds']),V/'reports/FEATURE_STATUS.json')
 print('V7 features READY',json.dumps(audit['checks']),flush=True)

if __name__=='__main__':
 if '--dpd-only' in sys.argv:
  a=pd.concat([pd.read_csv(RAW/'train_applications.csv'),pd.read_csv(RAW/'test_applications.csv')]);a['decision_datetime']=pd.to_datetime(a.decision_datetime);print(json.dumps(audit_dpd(a),indent=2))
 else:build()
