"""Rebuild V25 coverage and one allocation-opportunity extension from raw data."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,time,gc
import numpy as np
import pandas as pd
import baseline as b

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/mnt/home/logika-faris-exploration/exploration-12h-20261017/raw')
INPUT=Path('/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/input')
MONTHS=['risk_conditional_negative_months','risk_conditional_nonnegative_months']
AUX=['aux_account_months','aux_orders90','aux_ecommerce_observed_days','aux_allocation_count']
NEW=['coverage_no_allocation_negative_months','coverage_no_allocation_nonnegative_months',
     'coverage_mixed_allocation_negative_months','coverage_mixed_allocation_nonnegative_months']

def build():
    start=time.time();dest=ROOT/'data';dest.mkdir(exist_ok=True)
    assert not (dest/'AUDIT.json').exists(),'Do not overwrite completed features'
    frame=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id'])
    apps=pd.read_csv(RAW/'train_applications.csv',usecols=['application_id','customer_id','decision_datetime'],parse_dates=['decision_datetime'])
    apps=apps.set_index('application_id').loc[frame.application_id].reset_index()
    assert apps.customer_id.tolist()==frame.customer_id.tolist()
    apps['row']=np.arange(len(apps));lookup=apps[['customer_id','row','decision_datetime']]
    out=pd.DataFrame({'application_id':frame.application_id});n=len(out)
    source_files=[RAW/'train_applications.csv',INPUT/'data/train_features.parquet']
    def history(name,dates):
        path=RAW/(name+'.csv');source_files.append(path)
        d=pd.read_csv(path,parse_dates=dates)
        return d.merge(lookup,on='customer_id',how='inner',validate='many_to_one')
    def counts(series,missing=0):return series.reindex(np.arange(n),fill_value=missing).to_numpy()
    ac=history('account_behavior',['month'])
    assert not ac.duplicated(['row','month']).any()
    ac=ac[(ac.month+pd.offsets.MonthBegin(1))<=ac.decision_datetime].copy()
    ac['net']=ac.inflows-ac.outflows
    recent=ac.month>=ac.decision_datetime.dt.to_period('M').dt.to_timestamp()-pd.DateOffset(months=3)
    out[AUX[0]]=counts(ac[recent].groupby('row').size())
    repayment=history('repayment_history',['due_date','last_payment_datetime'])
    assert (repayment.last_payment_datetime.isna()|(repayment.last_payment_datetime<repayment.decision_datetime)).all()
    repayment=repayment[repayment.due_date<repayment.decision_datetime].copy()
    out[AUX[3]]=counts(repayment[repayment.last_payment_datetime.notna()].groupby('row').size())
    age=(repayment.decision_datetime.dt.normalize()-repayment.due_date.dt.normalize()).dt.days
    eligible=repayment[age>=15].copy()
    eligible['month']=eligible.due_date.dt.to_period('M').dt.to_timestamp()
    eligible['allocated']=eligible.last_payment_datetime.notna()
    monthly=eligible.groupby(['row','month']).allocated.agg(['sum','size']).reset_index()
    monthly=monthly.merge(ac[['row','month','net']],on=['row','month'],how='inner',validate='one_to_one')
    states={};assert (monthly['sum']<=monthly['size']).all()
    for condition,cf in [('negative',monthly.net<0),('nonnegative',monthly.net>=0)]:
        counts_for=lambda mask:counts(monthly[cf&mask].groupby('row').size())
        out['risk_conditional_'+condition+'_months']=counts_for(monthly['sum']>0)
        out['coverage_no_allocation_'+condition+'_months']=counts_for(monthly['sum']==0)
        out['coverage_mixed_allocation_'+condition+'_months']=counts_for((monthly['sum']>0)&(monthly['sum']<monthly['size']))
        total=counts_for(np.ones(len(monthly),bool))
        assert np.array_equal(total,out['risk_conditional_'+condition+'_months']+out['coverage_no_allocation_'+condition+'_months'])
        states[condition]=dict(eligible_months=int(total.sum()),recorded_allocation_months=int(out['risk_conditional_'+condition+'_months'].sum()),
            no_allocation_months=int(out['coverage_no_allocation_'+condition+'_months'].sum()),
            mixed_allocation_months=int(out['coverage_mixed_allocation_'+condition+'_months'].sum()))
    del eligible,repayment,monthly,ac;gc.collect()
    ec=history('ecommerce_transactions',['transaction_datetime'])
    assert (ec.transaction_datetime<ec.decision_datetime).all()
    assert ec.groupby('order_id').row.nunique().max()==1
    orders=ec.drop_duplicates('order_id').copy();orders['age']=(orders.decision_datetime-orders.transaction_datetime).dt.total_seconds()/86400
    complete=orders.order_status.eq('completed')&orders.age.between(0,90,inclusive='right')
    out[AUX[1]]=counts(orders[complete].groupby('row').size())
    out[AUX[2]]=counts(orders.groupby('row').age.max(),np.nan)
    # Check the independent raw rebuild against the teammate's exported columns.
    reference=pd.read_parquet(ROOT/'evidence/extension/data/raw_features.parquet').set_index('application_id').loc[frame.application_id]
    for condition in ['negative','nonnegative']:
        name='risk_conditional_'+condition+'_months';old='ext_aux_allocation_'+condition+'_n'
        assert np.array_equal(out[name].to_numpy(),reference[old].to_numpy()),name
    imported=Path('/mnt/home/logika-faris-exploration/teammate-approach-20261020/feature_evidence')
    comparisons=[]
    for f in range(5):
        ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz')
        for label in ['correction','validation']:
            part=pd.read_parquet(imported/f'data/fold{f}/{label}.parquet')
            rows=ix[label];assert part.application_id.tolist()==frame.application_id.iloc[rows].tolist()
            for name in AUX:
                assert np.array_equal(out[name].to_numpy()[rows],part[name].to_numpy(),equal_nan=True),(f,label,name)
            comparisons.append(dict(fold=f,partition=label,columns_verified=4,rows=len(rows)))
    out.to_parquet(dest/'features.parquet',index=False)
    audit=dict(status='passed',rows=n,seconds=time.time()-start,rebuilt_V25_columns=MONTHS+AUX,new_columns=NEW,
        source_hashes={str(path):b.sha(path) for path in source_files},feature_sha=b.sha(dest/'features.parquet'),
        matched_reference=b.sha(ROOT/'evidence/extension/data/raw_features.parquet'),support_rebuild_checks=comparisons,
        monthly_partition=states,selected_default_labels_loaded=False,new_max_DPD_fields_used=False,
        event_semantics='Recorded allocation timestamp is not proof of full settlement. Mixed months include both allocated and unallocated installments.',
        missing_semantics='Zero overlap counts mean no observed eligible overlap; they do not assert low default risk.',
        point_in_time_qualification='Own timestamps and complete account months checked; original baseline DPD snapshot qualification inherited.')
    b.dump(audit,dest/'AUDIT.json');print(json.dumps(audit),flush=True)

if __name__=='__main__':build()
