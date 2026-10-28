"""Construct interval/right-censored historical outcomes and origination-time inputs."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import numpy as np
import pandas as pd
from common import R,root_path,dump,sha,utc
DAY=pd.Timedelta(days=1)
CONTRACT=['principal','rate','term','installment']
def contract(d,historical=False):
    names=['approved_amount','annual_interest_rate','term_months','scheduled_monthly_installment'] if historical else ['loan_amount','annual_interest_rate','term_months','proposed_monthly_installment']
    return pd.DataFrame({k:d[n].to_numpy(float) for k,n in zip(CONTRACT,names)})
def attach_bureau(apps,table):
    wanted=['score','active_accounts','outstanding_balance','utilization','inquiries_last_6m','delinquent_accounts_last_12m']
    b=table.merge(apps[['application_id','decision_datetime']],on='application_id',how='inner',validate='many_to_one')
    b=b[(b.available_datetime< b.decision_datetime)&(b.report_datetime<b.decision_datetime)]
    b=b.sort_values(['available_datetime','report_datetime','bureau_observation_id']).drop_duplicates(['application_id','bureau_provider'],keep='last')
    # Overlapping provider debt totals are averaged, never added.
    return b.groupby('application_id')[wanted].mean().add_prefix('bureau_').reindex(apps.application_id).reset_index(drop=True)
def main():
    raw=root_path('raw');original=pd.read_parquet(root_path('packet_root')/'packets/fold0/current.parquet')
    train=pd.read_csv(raw/'train_applications.csv',parse_dates=['decision_datetime'])
    test=pd.read_csv(raw/'test_applications.csv',parse_dates=['decision_datetime'])
    bank_meta=pd.read_parquet(root_path('bank')/'metadata.parquet')
    train=train.set_index('application_id').loc[bank_meta.application_id[:70000]].reset_index()
    assert len(train)==70000 and train.customer_id.is_unique
    assert set(train.customer_id).isdisjoint(test.customer_id)
    allapps=pd.concat([train.drop(columns='target'),test],ignore_index=True)
    history=pd.read_csv(raw/'previous_applications.csv',parse_dates=['decision_datetime'])
    history=history[history.decision.eq('approved')&history.loan_id.notna()].copy()
    history=history.merge(allapps[['customer_id','decision_datetime']].rename(columns={'decision_datetime':'cutoff'}),on='customer_id',validate='many_to_one')
    assert history.loan_id.is_unique and (history.decision_datetime<history.cutoff).all()
    rep=pd.read_csv(raw/'repayment_history.csv',parse_dates=['due_date','last_payment_datetime'])
    assert not rep.installment_id.duplicated().any()
    rep=rep.merge(history[['loan_id','customer_id','decision_datetime','cutoff']],on=['loan_id','customer_id'],validate='many_to_one')
    assert (rep.due_date<rep.cutoff).all()
    assert (rep.last_payment_datetime.isna()|(rep.last_payment_datetime<rep.cutoff)).all()
    age=(rep.cutoff-rep.due_date)/DAY
    assert (rep.max_days_past_due<=np.ceil(age)).all() and (rep.days_past_due<=np.ceil(age)).all()
    event=rep.max_days_past_due.ge(90)
    rep['event_lower']=rep.due_date+pd.to_timedelta(90,unit='D')
    # This upper bound is conservative; neither maximum DPD nor last allocation is an exact event date.
    # Snapshot maximum implies DPD90 occurred by cutoff. For fully-paid installments it also occurred by the final allocation.
    rep['event_upper']=rep.cutoff
    fully_paid=rep.amount_paid.ge(rep.amount_due)&rep.last_payment_datetime.notna()
    rep.loc[fully_paid,'event_upper']=rep.loc[fully_paid,'last_payment_datetime']
    valid_event=event&(rep.event_upper>=rep.event_lower)
    invalid=int((event&~valid_event).sum());assert invalid==0,invalid
    events=rep[valid_event].groupby('loan_id').agg(lower=('event_lower','min'),upper=('event_upper','min'))
    observed=rep.groupby('loan_id').agg(installments=('installment_id','size'),first_due=('due_date','min'),last_due=('due_date','max'))
    history=history.join(events,on='loan_id').join(observed,on='loan_id')
    history['event']=history.lower.notna()
    history['age_days']=(history.cutoff-history.decision_datetime)/DAY
    history['lower_days']=((history.lower-history.decision_datetime)/DAY).where(history.event,history.age_days)
    history['upper_days']=((history.upper-history.decision_datetime)/DAY).where(history.event,np.inf)
    assert (history.lower_days>=0).all() and (history.loc[history.event,'upper_days']>=history.loc[history.event,'lower_days']).all()
    # Every installment whose monthly due date is already observable should be represented.
    expected=np.zeros(len(history),int)
    for month in range(1,int(history.term_months.max())+1):
        due=history.decision_datetime+pd.DateOffset(months=month)
        expected+=(history.term_months.ge(month)&due.lt(history.cutoff)).to_numpy(int)
    history['expected_installments']=expected
    history['schedule_complete']=history.installments.fillna(0).ge(expected)&history.installments.notna()
    horizon=history.decision_datetime+pd.DateOffset(months=12)
    positive=history.event&history.upper.le(horizon)
    negative=history.cutoff.ge(horizon)&(~history.event|history.lower.gt(horizon))&history.schedule_complete
    history['target12']=np.where(positive,1,np.where(negative,0,-1))
    # Survival likelihood uses interval-censored positives and right-censored no-event loans, without converting short observation into a negative label.
    historical_features=contract(history,True)
    bureau=pd.read_csv(raw/'bureau_observations.csv',parse_dates=['report_datetime','available_datetime'])
    historical_features=pd.concat([historical_features,attach_bureau(history,bureau)],axis=1)
    current_features=pd.concat([contract(allapps),attach_bureau(allapps,bureau)],axis=1)
    # Bank history is joined only from completed months preceding each loan's own decision.
    account=pd.read_csv(raw/'account_behavior.csv',parse_dates=['month'])
    def attach_account(apps):
        z=account.merge(apps[['application_id','customer_id','decision_datetime']],on='customer_id',validate='many_to_many')
        z=z[(z.month+pd.offsets.MonthBegin(1)<=z.decision_datetime)&(z.month>=z.decision_datetime-pd.DateOffset(months=3))]
        return z.groupby('application_id')[['income_like_credits','outflows','average_balance','low_balance_days']].mean().add_prefix('account_').reindex(apps.application_id).reset_index(drop=True)
    historical_features=pd.concat([historical_features,attach_account(history)],axis=1)
    current_features=pd.concat([current_features,attach_account(allapps)],axis=1)
    assert historical_features.columns.tolist()==current_features.columns.tolist()
    identifiers=['application_id','customer_id','loan_id','decision_datetime','cutoff','age_days','lower_days','upper_days','event','schedule_complete','target12','expected_installments','installments']
    destination=R/'prepared';destination.mkdir(exist_ok=False)
    pd.concat([history[identifiers].reset_index(drop=True),historical_features],axis=1).to_parquet(destination/'historical.parquet',index=False)
    pd.concat([allapps[['application_id','customer_id']].reset_index(drop=True),current_features],axis=1).to_parquet(destination/'current.parquet',index=False)
    train[['application_id','customer_id','target']].to_parquet(destination/'labels_private.parquet',index=False)
    audit={'status':'passed_for_interval_and_censoring_model','UTC':utc(),'historical_approved':len(history),
           'age_at_least365':int(history.age_days.ge(365).sum()),'customers':history.customer_id.nunique(),
           'interval_events':int(history.event.sum()),'right_censored':int((~history.event).sum()),
           'schedule_complete':int(history.schedule_complete.sum()),'target12_counts':history.target12.value_counts().sort_index().to_dict(),
           'invalid_event_bounds':invalid,'exact_first_DPD90_dates_available':False,
           'teacher_features':current_features.columns.tolist(),'profile_at_historical_origination_used':False,
           'positive_bound':'first qualifying due+90 <= first loan90 event <= earliest final allocation among paid late installments, or current selected cutoff; conservative interval',
           'negative_rule':'12 calendar months fully observed and schedule complete; no qualifying event or event lower bound strictly after12m',
           'censoring_assumptions':'Repayment snapshot contains all originated installments due before selected cutoff and complete lifetime maximum to cutoff. Schedule coverage audited; exact upstream extraction remains an assumption.',
           'raw_sha256':{n:sha(raw/n) for n in ['train_applications.csv','test_applications.csv','previous_applications.csv','repayment_history.csv','bureau_observations.csv','account_behavior.csv']},
           'prepared_sha256':{p.name:sha(p) for p in destination.glob('*.parquet')},'source_sha256':sha(__file__)}
    dump(audit,R/'SHADOW_DATA_GATE.json');print(__import__('json').dumps(audit,default=str),flush=True)
if __name__=='__main__':main()
