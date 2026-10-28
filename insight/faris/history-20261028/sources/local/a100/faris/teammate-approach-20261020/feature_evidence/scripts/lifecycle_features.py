"""First observed payments, MOB benchmarks, auxiliary residuals and information.

No selected-loan default labels are used. Historical repayment/approval outcomes
are separate auxiliary responses. References exclude the query inner customer
fold and customers whose selected snapshot cutoff is not before query month.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import time,argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from features import ROOT,V,RAW,div
from baseline import dump,sha,ids_hash
from experiments import combined

L=V/'lifecycle'
DEFINITIONS=[
 ('first_observed_max_dpd','first_payment','mean(maxDPD of earliest observed installment per historical loan), due-age>=30'),
 ('first3_dpd15_rate','first_payment','loan-equal mean(1[maxDPD>=15] among first3 observed installments with due-age>=15)'),
 ('early_failure_frequency','first_payment','share eligible loans with first observed installment maxDPD>=30, due-age>=30'),
 ('early_to_late_dpd_ratio','first_payment','mean(maxDPD first3 observed installments)/mean(maxDPD later observed installments); both due-age>=30; denominator0 -> missing'),
 ('first_payment_shortfall','first_payment','mean(max(amount_due-amount_paid,0)/amount_due on first observed installment), due-age>=30'),
 ('first_payment_eligible_loans','first_payment','number historical loans with first observed installment due-age>=30'),
 ('mob_adjusted_dpd15','seasoning','loan-equal mean(y15-cell_expected15); cell=(MOBbin,tenor,vintage_halfyear,observation_agebin); weighted cell prior20 toward dated population mean'),
 ('mob_adjusted_dpd30','seasoning','same y30-cell_expected30 among due-age>=30'),
 ('mob_delinquency_surprise','seasoning','loan-equal mean(-y15*log(expected15)-(1-y15)*log(1-expected15))'),
 ('mob_early_risk_excess','seasoning','loan-equal mean(y15-expected15), approval-decision-proxy MOB at due<=3'),
 ('mob_expected_vs_actual_severity','seasoning','loan-equal mean(min(maxDPD,90)/90-cell_expected_capped_severity)'),
 ('repayment_excess_risk_mean','repayment_residual','loan-equal mean(y15-auxiliary_predicted_p15)'),
 ('repayment_excess_risk_p90','repayment_residual','P90(y15-auxiliary_predicted_p15), observed eligible installments'),
 ('repayment_unexpected_good_payment_rate','repayment_residual','loan-equal mean(1[y15=0 and predicted_p15>=0.5]); fixed threshold'),
 ('repayment_unexpected_bad_payment_rate','repayment_residual','loan-equal mean(1[y15=1 and predicted_p15<0.15]); fixed threshold'),
 ('repayment_residual_uncertainty','repayment_residual','sqrt(loan-equal mean(p15*(1-p15))/number observed eligible loans); noise proxy, not cluster-robust CI'),
 ('lifecycle_early_dpd15','lifecycle','loan-equal DPD15 rate in phase MOB/tenor<=1/3; due-age>=15'),
 ('lifecycle_middle_dpd15','lifecycle','loan-equal DPD15 rate in phase1/3<MOB/tenor<=2/3; due-age>=15'),
 ('lifecycle_late_dpd15','lifecycle','loan-equal DPD15 rate in phase MOB/tenor>2/3; due-age>=15'),
 ('lifecycle_late_minus_early','lifecycle','within-loan late-phase DPD15 rate minus early-phase rate; mean only loans with both phases observed'),
 ('lifecycle_severity_slope','lifecycle','mean within-loan OLS slope(min(maxDPD,90)/90 versus MOB/tenor), >=3 eligible installments and distinct MOB; due-age>=30'),
 ('thin_source_signature','thin_file','categorical combination(previous yes/no, repayment yes/no, bureau absent/unscored/scored, account yes/no); no ordinal pattern coding'),
 ('thin_oldest_internal_loan_days','thin_file','selected decision minus earliest known approved historical decision; observation-age proxy'),
 ('thin_matured_installment_count','thin_file','count historical installments with due-age>=30'),
 ('thin_repayment_calendar_density','thin_file','number unique observed due months/(last due month-first due month+1)'),
 ('thin_bureau_reports_per_month','thin_file','available bureau report count/max(1,span between first and last available reports in30-day units)'),
 ('thin_account_calendar_coverage','thin_file','complete account months observed/(last observed month-first observed month+1)'),
 ('bureau_observed_debt_low_proxy','bureau_ambiguity','max latest-available provider outstanding_balance; observed-coverage scenario only'),
 ('bureau_observed_debt_high_proxy','bureau_ambiguity','sum latest-available provider outstanding_balance; overlapping-account scenario only, not identified total debt'),
 ('bureau_exposure_ambiguity_gap','bureau_ambiguity','sum latest provider debt minus max latest provider debt'),
 ('bureau_exposure_relative_gap','bureau_ambiguity','(sum latest provider debt-max latest provider debt)/max latest provider debt; zero denominator undefined'),
 ('bureau_affordability_low_proxy','bureau_ambiguity','max latest provider debt/selected declared monthly income'),
 ('bureau_affordability_high_proxy','bureau_ambiguity','sum latest provider debt/selected declared monthly income'),
 ('bureau_exposure_provider_count','bureau_ambiguity','number latest-available provider reports contributing to observed debt scenarios'),
 ('approval_propensity_mean','approval_propensity','mean predicted approval probability on eligible historical approved/rejected applications'),
 ('approval_observed_expected_gap','approval_propensity','mean(1[historically approved]-auxiliary_predicted_approval_probability)'),
 ('approval_unexpected_rejection_rate','approval_propensity','mean(1[historically rejected and predicted_approval>=0.5]); fixed threshold'),
 ('approval_residual_uncertainty','approval_propensity','sqrt(sum(p*(1-p)))/number historical approved/rejected observations; conditional noise proxy'),
 ('approval_propensity_observations','approval_propensity','number eligible historical approved/rejected applications; cancelled excluded and separately logged')]
FITTED={'seasoning','repayment_residual','approval_propensity'}
RCOLS=['log_amount','apr','term','mob_fraction','log_due_age','vintage','due_sin','due_cos']
ACOLS=['log_requested','term','application_period','application_sin','application_cos']
CELL=['mob_bin','term','vintage_halfyear','age_bin']

def catalog():
 rows=[]
 for name,family,formula in DEFINITIONS:
  rows.append(dict(feature_name='life_'+name,family=family,formula=formula,status='FOLD_LOCAL' if family in FITTED else 'IMPLEMENTED',
   source='historical applications/loans, repayment, bureau, complete account observations; selected cutoff/application metadata',
   cutoff_rule='Own historical decision/due before selected cutoff; threshold-matured installments; provider available<=cutoff; references from correction customers with selected snapshot before query decision-month start; three inner customer folds',
   leakage_risk='Unverified DPD/paid snapshot provenance remains documented; DPD allowed without quarantine' if family in {'first_payment','seasoning','repayment_residual','lifecycle'} else 'Incrementally low; inherited baseline uncertainty',
   caveat='Earliest observed installment need not be contractual first; historical approval decision is an origination proxy; no causal or identified true debt claim. Auxiliary benchmarks are computed as of selected cutoff, not reconstructed old underwriting scores.',validation_status='Unvalidated',priority='P1'))
 return pd.DataFrame(rows)

def assign(frame,name,values):frame['life_'+name]=np.asarray(values)

def equal_loan_mean(events,values):
 d=events[['row','loan_id']].copy();d['value']=np.asarray(values)
 return d.groupby(['row','loan_id']).value.mean().groupby('row').mean()

def add_positions(r):
 r=r.sort_values(['row','loan_id','due_date','installment_id'],kind='stable').copy()
 # Opaque installment_id only breaks an exact date tie; never a numeric feature.
 assert not r.duplicated(['loan_id','due_date']).any(),'Ambiguous within-loan due order requires explicit resolution'
 r['observed_position']=r.groupby(['row','loan_id']).cumcount()+1
 r['mob']=(r.due_date.dt.year-r.origin.dt.year)*12+r.due_date.dt.month-r.origin.dt.month
 assert (r.mob>=0).all();r['mob_fraction']=div(r.mob,r.term)
 return r

def build():
 started=time.time();(L/'data').mkdir(parents=True,exist_ok=True);(L/'reports').mkdir(exist_ok=True)
 a=pd.read_parquet(V/'discovery/data/applications.parquet');a['decision_datetime']=pd.to_datetime(a.decision_datetime);a.index.name='row'
 assert a.application_id.is_unique and a.customer_id.is_unique and 'target' not in a
 lookup=a.reset_index()[['row','customer_id','decision_datetime']]
 f=pd.DataFrame(dict(application_id=a.application_id,life_aux_row=np.arange(len(a)),life_aux_decision_month=a.decision_datetime.dt.to_period('M').astype(int)),index=a.index)
 align=lambda s,fill=np.nan:s.reindex(f.index,fill_value=fill)
 prev=pd.read_csv(RAW/'previous_applications.csv',engine='pyarrow');prev['origin']=pd.to_datetime(prev.pop('decision_datetime'));prev['application_datetime']=pd.to_datetime(prev.application_datetime)
 prev=prev.merge(lookup,on='customer_id',validate='many_to_one');prev=prev[(prev.origin<prev.decision_datetime)&~prev.application_id.isin(a.application_id)].copy()
 loan=prev[(prev.decision=='approved')&prev.loan_id.notna()].copy();assert loan.loan_id.is_unique
 info=loan[['row','customer_id','loan_id','origin','decision_datetime','term_months','annual_interest_rate']].rename(columns={'term_months':'term','annual_interest_rate':'apr'})
 r=pd.read_csv(RAW/'repayment_history.csv',engine='pyarrow');original=len(r);assert r.installment_id.is_unique;r['due_date']=pd.to_datetime(r.due_date)
 r=r.merge(info,on=['customer_id','loan_id'],validate='many_to_one');assert len(r)==original and r.row.notna().all()
 r=r[r.due_date<r.decision_datetime].copy();r['age']=(r.decision_datetime.dt.normalize()-r.due_date.dt.normalize()).dt.days
 r=add_positions(r);r['y15']=(r.max_days_past_due>=15).astype(int);r['y30']=(r.max_days_past_due>=30).astype(int);r['severity']=r.max_days_past_due.clip(upper=90)/90
 first=r[(r.observed_position==1)&(r.age>=30)]
 assign(f,'first_observed_max_dpd',align(first.groupby('row').max_days_past_due.mean()))
 early=r[(r.observed_position<=3)&(r.age>=15)]
 assign(f,'first3_dpd15_rate',align(equal_loan_mean(early,early.y15)))
 assign(f,'early_failure_frequency',align(first.groupby('row').y30.mean()))
 early30=r[(r.observed_position<=3)&(r.age>=30)];late30=r[(r.observed_position>3)&(r.age>=30)]
 assign(f,'early_to_late_dpd_ratio',div(align(equal_loan_mean(early30,early30.max_days_past_due)),align(equal_loan_mean(late30,late30.max_days_past_due))))
 assign(f,'first_payment_shortfall',align(first.assign(shortfall=div(np.maximum(first.amount_due-first.amount_paid,0),first.amount_due)).groupby('row').shortfall.mean()))
 assign(f,'first_payment_eligible_loans',align(first.groupby('row').loan_id.nunique(),0))
 mature=r[r.age>=15].copy();mature['phase']=np.select([mature.mob_fraction<=1/3,mature.mob_fraction<=2/3],[0,1],default=2)
 phase=mature.groupby(['row','loan_id','phase']).y15.mean().unstack('phase').reindex(columns=[0,1,2])
 for number,name in [(0,'early'),(1,'middle'),(2,'late')]:assign(f,'lifecycle_'+name+'_dpd15',align(phase[number].groupby('row').mean()))
 assign(f,'lifecycle_late_minus_early',align((phase[2]-phase[0]).dropna().groupby('row').mean()))
 q=r[r.age>=30].copy();q['xx']=q.mob_fraction**2;q['xy']=q.mob_fraction*q.severity
 st=q.groupby(['row','loan_id']).agg(n=('row','size'),x=('mob_fraction','sum'),y=('severity','sum'),xx=('xx','sum'),xy=('xy','sum'),distinct=('mob','nunique'))
 slope=div(st.xy-st.x*st.y/st.n,st.xx-st.x**2/st.n);slope=pd.Series(slope,index=st.index).where((st.n>=3)&(st.distinct>=2))
 assign(f,'lifecycle_severity_slope',align(slope.groupby('row').mean()))
 print('Lifecycle first-observed payment and phase features READY',flush=True)
 # Availability signature: genuinely absent source remains distinct from scored/unscored.
 bu=pd.read_csv(RAW/'bureau_observations.csv',engine='pyarrow');bu['available_datetime']=pd.to_datetime(bu.available_datetime);bu['report_datetime']=pd.to_datetime(bu.report_datetime)
 bu=bu.merge(lookup,on='customer_id',validate='many_to_one');bu=bu[(bu.available_datetime<=bu.decision_datetime)&(bu.report_datetime<=bu.available_datetime)].copy()
 last=bu.sort_values(['row','bureau_provider','report_datetime','available_datetime']).groupby(['row','bureau_provider']).tail(1)
 ac=pd.read_csv(RAW/'account_behavior.csv',usecols=['customer_id','month'],engine='pyarrow');ac['month']=pd.to_datetime(ac.month);ac=ac.merge(lookup,on='customer_id',validate='many_to_one');ac=ac[ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime]
 hasprev=align(prev.groupby('row').size(),0)>0;hasrep=align(r.groupby('row').size(),0)>0;hasbureau=align(bu.groupby('row').size(),0)>0;scored=align(last.groupby('row').score.count(),0)>0;hasaccount=align(ac.groupby('row').size(),0)>0
 f['life_thin_source_signature']=['prev%d|rep%d|bureau_%s|account%d'%(p,h,'scored' if s else ('unscored' if b else 'absent'),acv) for p,h,b,s,acv in zip(hasprev,hasrep,hasbureau,scored,hasaccount)]
 assign(f,'thin_oldest_internal_loan_days',(a.decision_datetime-align(loan.groupby('row').origin.min())).dt.total_seconds()/86400)
 assign(f,'thin_matured_installment_count',align(r[r.age>=30].groupby('row').size(),0))
 r['due_period']=r.due_date.dt.to_period('M').astype(int);rs=r.groupby('row').due_period.agg(['nunique','min','max'])
 assign(f,'thin_repayment_calendar_density',div(align(rs['nunique']),align(rs['max']-rs['min']+1)))
 bs=bu.groupby('row').available_datetime.agg(['size','min','max']);span=(bs['max']-bs['min']).dt.total_seconds()/86400/30
 assign(f,'thin_bureau_reports_per_month',div(align(bs['size']),align(span.clip(lower=1))))
 ac['period']=ac.month.dt.to_period('M').astype(int);acs=ac.groupby('row').period.agg(['nunique','min','max'])
 assign(f,'thin_account_calendar_coverage',div(align(acs['nunique']),align(acs['max']-acs['min']+1)))
 debt=last.groupby('row').outstanding_balance.agg(['max','sum','size']);low=align(debt['max']);high=align(debt['sum']);gap=high-low
 for name,value in [('observed_debt_low_proxy',low),('observed_debt_high_proxy',high),('exposure_ambiguity_gap',gap),('exposure_relative_gap',div(gap,low)),('affordability_low_proxy',div(low,a.declared_monthly_income)),('affordability_high_proxy',div(high,a.declared_monthly_income)),('exposure_provider_count',align(debt['size'],0))]:assign(f,'bureau_'+name,value)
 print('Lifecycle thin-file and bureau scenario features READY',flush=True)
 # Exact contract inputs existed before historical payments; current profiles excluded.
 mature['log_amount']=np.log1p(mature.amount_due);mature['log_due_age']=np.log1p(mature.age)
 mature['vintage']=mature.origin.dt.to_period('M').astype(int);mature['vintage_halfyear']=mature.origin.dt.year*2+(mature.origin.dt.month>6).astype(int)
 mature['due_sin']=np.sin(2*np.pi*mature.due_date.dt.month/12);mature['due_cos']=np.cos(2*np.pi*mature.due_date.dt.month/12)
 mature['mob_bin']=np.digitize(mature.mob,[3,6,9,12],right=True);mature['age_bin']=np.digitize(mature.age,[30,60,90,180],right=True)
 mature['weight']=1/mature.groupby(['row','loan_id']).row.transform('size')
 keep=['row','loan_id','age','mob','y15','y30','severity','weight']+list(dict.fromkeys(RCOLS+CELL))
 mature[keep].to_parquet(L/'data/repayment_events.parquet',index=False)
 approval=prev[prev.decision.isin(['approved','rejected'])].copy();approval['approved']=(approval.decision=='approved').astype(int)
 approval['log_requested']=np.log1p(approval.requested_amount);approval['term']=approval.term_months;approval['application_period']=approval.application_datetime.dt.to_period('M').astype(int)
 approval['application_sin']=np.sin(2*np.pi*approval.application_datetime.dt.month/12);approval['application_cos']=np.cos(2*np.pi*approval.application_datetime.dt.month/12)
 approval[['row','approved']+ACOLS].to_parquet(L/'data/approval_events.parquet',index=False)
 for name,family,_ in DEFINITIONS:
  if family in FITTED and 'life_'+name not in f:f['life_'+name]=np.nan
 assign(f,'approval_propensity_observations',align(approval.groupby('row').size(),0))
 for col in f.columns.difference(['application_id','life_thin_source_signature']):f[col]=pd.to_numeric(f[col]).replace([np.inf,-np.inf],np.nan)
 f.to_parquet(L/'data/raw_features.parquet',index=False);catalog().to_csv(L/'reports/FEATURE_CATALOG.csv',index=False)
 dump(dict(status='READY',rows=len(f),catalog_features=len(DEFINITIONS),feature_sha256=sha(L/'data/raw_features.parquet'),repayment_events_sha256=sha(L/'data/repayment_events.parquet'),approval_events_sha256=sha(L/'data/approval_events.parquet'),
  elapsed_seconds=time.time()-started,first_observed_not_certified_contractual_first=True,origin_uses_historical_approval_decision_proxy=True,DPD_allowed=True,snapshot_provenance_verified=False,
  auxiliary_predictors_exclude_selected_profile_and_approval_outputs=True,no_selected_default_labels=True,cancelled_historical_requests_excluded_from_approval=int((prev.decision=='cancelled').sum())),L/'reports/FEATURE_STATUS.json')
 print('Lifecycle raw features READY',flush=True)

def fit_classifier(reference,columns,target,weight=None):
 assert 'target' not in columns and target!='target'
 y=reference[target].to_numpy(int);w=np.ones(len(y)) if weight is None else reference[weight].to_numpy(float)
 probability=float((np.dot(w,y)+.5)/(w.sum()+1)) if len(y) else (.5 if target=='approved' else .15)
 if len(y)<100 or len(np.unique(y))<2:return dict(constant=probability,rows=len(y))
 model=HistGradientBoostingClassifier(max_iter=50,max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=20.,learning_rate=.05,early_stopping=False,random_state=31027)
 model.fit(reference[columns],y,sample_weight=w)
 return dict(model=model,rows=len(y),columns=columns)

def predict(model,frame,columns):
 if not len(frame):return np.array([],dtype=float)
 return np.full(len(frame),model['constant']) if 'constant' in model else model['model'].predict_proba(frame[columns])[:,1]

def seasonal_reference(reference,target,eligible):
 d=reference[eligible].copy();d['k']=d[target]*d.weight
 den=d.weight.sum();mu=float((d.k.sum()+.5)/(den+1)) if len(d) else .15
 cells=d.groupby(CELL).agg(k=('k','sum'),n=('weight','sum'));cells['expected']=(cells.k+20*mu)/(cells.n+20)
 return mu,cells.expected

def seasonal_expected(frame,reference):
 mu,table=reference
 if len(frame)==0:return np.array([])
 return pd.MultiIndex.from_frame(frame[CELL]).map(table).to_series(index=frame.index).fillna(mu).to_numpy(float)

def populate(query,reference,repay,approvals,ledger,cache):
 for month,positions in query.groupby('life_aux_decision_month').groups.items():
  positions=list(positions);q=query.loc[positions];ref=reference[reference.life_aux_decision_month<month]
  ids=ref.life_aux_row.to_numpy(int);key=(tuple(ids),int(month))
  if key not in cache:
   rr=repay[repay.row.isin(ids)];aa=approvals[approvals.row.isin(ids)]
   model=fit_classifier(rr,RCOLS,'y15','weight');am=fit_classifier(aa,ACOLS,'approved')
   seasonal={label:seasonal_reference(rr,target,rr.age>=age) for label,target,age in [('15','y15',15),('30','y30',30),('severity','severity',30)]}
   cache[key]=(model,am,seasonal);ledger.append(dict(month=int(month),reference_customer_ids=ids,repayment_model=model,approval_model=am,seasoning=seasonal))
  model,am,seasonal=cache[key];events=repay[repay.row.isin(q.life_aux_row)].copy();mapping=q.set_index('life_aux_row').index
  def put(name,series):query.loc[positions,'life_'+name]=q.life_aux_row.map(series).to_numpy()
  for label,threshold in [('15',15),('30',30)]:
   d=events[events.age>=threshold].copy();expected=seasonal_expected(d,seasonal[label]);residual=d['y'+label].to_numpy()-expected
   put('mob_adjusted_dpd'+label,equal_loan_mean(d,residual))
   if label=='15':
    p=np.clip(expected,1e-6,1-1e-6);put('mob_delinquency_surprise',equal_loan_mean(d,-d.y15*np.log(p)-(1-d.y15)*np.log(1-p)))
    early=d.mob<=3;put('mob_early_risk_excess',equal_loan_mean(d[early],residual[early.to_numpy()]))
  d=events[events.age>=30];put('mob_expected_vs_actual_severity',equal_loan_mean(d,d.severity.to_numpy()-seasonal_expected(d,seasonal['severity'])))
  p=predict(model,events,RCOLS);residual=events.y15.to_numpy()-p;put('repayment_excess_risk_mean',equal_loan_mean(events,residual))
  d=events[['row']].copy();d['residual']=residual;put('repayment_excess_risk_p90',d.groupby('row').residual.quantile(.9))
  put('repayment_unexpected_good_payment_rate',equal_loan_mean(events,(events.y15.to_numpy()==0)&(p>=.5)))
  put('repayment_unexpected_bad_payment_rate',equal_loan_mean(events,(events.y15.to_numpy()==1)&(p<.15)))
  noise=equal_loan_mean(events,p*(1-p));loans=events.groupby('row').loan_id.nunique();put('repayment_residual_uncertainty',np.sqrt(noise/loans))
  ap=approvals[approvals.row.isin(q.life_aux_row)].copy();p=predict(am,ap,ACOLS)
  ap['p']=p;ap['residual']=ap.approved-p;ap['unexpected']=(ap.approved==0)&(p>=.5);ap['variance']=p*(1-p)
  grouped=ap.groupby('row');put('approval_propensity_mean',grouped.p.mean());put('approval_observed_expected_gap',grouped.residual.mean());put('approval_unexpected_rejection_rate',grouped.unexpected.mean())
  put('approval_residual_uncertainty',np.sqrt(grouped.variance.sum())/grouped.size())
 return query

def fold(number):
 folder=L/'data'/f'fold{number}';folder.mkdir(parents=True,exist_ok=True);raw=pd.read_parquet(L/'data/raw_features.parquet')
 tr=pd.read_parquet(ROOT/'data/train_features.parquet');te=pd.read_parquet(ROOT/'data/test_features.parquet');idx=np.load(ROOT/'artifacts'/f'fold{number}'/'indices.npz');hi,vi=idx['correction'],idx['validation']
 assert len(hi)==6000 and len(vi)==14000 and not np.intersect1d(hi,vi).size;parts={}
 for name,base in [('correction',tr.iloc[hi]),('validation',tr.iloc[vi]),('test',te)]:
  old=pd.read_parquet(V/'extension/data'/f'fold{number}'/(name+'.parquet'));parts[name]=combined(base,old).merge(raw,on='application_id',how='left',validate='one_to_one')
 h=parts['correction'];repay=pd.read_parquet(L/'data/repayment_events.parquet');approval=pd.read_parquet(L/'data/approval_events.parquet');ledger=[];cache={}
 columns=catalog().query("status=='FOLD_LOCAL'").feature_name.tolist()
 for ti,qi in KFold(3,shuffle=True,random_state=31027+number).split(h):
  assert set(h.iloc[ti].life_aux_row).isdisjoint(h.iloc[qi].life_aux_row)
  q=populate(h.iloc[qi].copy(),h.iloc[ti],repay,approval,ledger,cache);h.loc[q.index,columns]=q[columns]
 for name in ['validation','test']:parts[name]=populate(parts[name],h,repay,approval,ledger,cache)
 for name,d in parts.items():
  keep=['application_id']+[c for c in d if c.startswith(('v7_','v7q_','aux_','ext_aux_','risk_','riskq_','life_'))];d[keep].to_parquet(folder/(name+'.parquet'),index=False)
 joblib.dump(dict(references=ledger,correction_customer_ids=h.application_id.tolist()),folder/'auxiliary_models.joblib',compress=3)
 dump(dict(status='PASS',fold=number,correction_ids_sha256=ids_hash(tr,hi),validation_ids_sha256=ids_hash(tr,vi),query_customer_crossfit=True,
  reference_snapshots_strictly_before_query_month=True,no_selected_default_labels_in_auxiliaries=True,approval_inputs=ACOLS,repayment_inputs=RCOLS,
  legacy_source_provenance_unverified=True,raw_sha256=sha(L/'data/raw_features.parquet')),folder/'AUDIT.json');print('Lifecycle fold READY',number,flush=True)

def checks():
 d=pd.DataFrame({'row':[0,0,1],'loan_id':['a','a','b'],'due_date':pd.to_datetime(['2024-02-01','2024-03-01','2024-06-01']),
  'origin':pd.to_datetime(['2024-01-01']*2+['2024-05-01']),'installment_id':['opaque_z','opaque_a','opaque_b'],'term':[12,12,6]})
 s=add_positions(d);assert s.observed_position.tolist()==[1,2,1] and s.mob.tolist()==[1,2,1]
 assert np.isclose(equal_loan_mean(pd.DataFrame({'row':[0,0,0],'loan_id':['a','a','b']}),[0,0,1]).iloc[0],.5)
 ref=pd.DataFrame({'mob_bin':[0,0],'term':[12,12],'vintage_halfyear':[4048,4048],'age_bin':[1,1],'y15':[1,0],'weight':[1.,1.],'age':[30,30]})
 expected=seasonal_expected(ref,seasonal_reference(ref,'y15',ref.age>=15));assert np.allclose(expected,.5)
 empty=pd.DataFrame(columns=['y15']+RCOLS+['weight']);assert fit_classifier(empty,RCOLS,'y15','weight')['constant']==.15
 assert not set(['approved_amount','loan_id','annual_interest_rate','target','declared_monthly_income']).intersection(ACOLS)
 assert np.isnan(div(1,0));assert {x[0] for x in DEFINITIONS}.__len__()==len(DEFINITIONS)
 (L/'reports').mkdir(parents=True,exist_ok=True);dump(dict(status='PASS',source_sha256=sha(__file__),checks=['date_order_not_identifier_order','MOB_not_due_age','equal_loan_exposure','smoothed_cell_expectation','empty_dated_reference_fallback','approval_outcome_inputs_excluded','undefined_ratio_missing','unique_catalog_names']),L/'reports/UNIT_CHECKS.json')
 print('Lifecycle integrity checks PASS',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int);p.add_argument('--checks',action='store_true');p.add_argument('--catalog',action='store_true');args=p.parse_args()
 if args.catalog:
  (L/'reports').mkdir(parents=True,exist_ok=True);catalog().to_csv(L/'reports/FEATURE_CATALOG.csv',index=False)
 elif args.checks:checks()
 elif args.fold is not None:fold(args.fold)
 else:build()
