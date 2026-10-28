"""Export labeled-customer raw sufficient statistics, without population fitting."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,hashlib,time,gc
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RAW=Path('/mnt/home/logika-faris-exploration/exploration-12h-20261017/raw')
INPUT=Path('/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/input')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
COV=['age','employment_tenure_months','dependents','proposed_monthly_installment',
     'annual_interest_rate','v2_existing_total_burden_observed',
     'v2_account_3m_minimum_balance_min','ecom_last_purchase_recency_days',
     'v2_ecom_completed_per_observed_month','has_account','has_repayments']

def run():
    start=time.time();dest=ROOT/'input';dest.mkdir(exist_ok=True)
    assert not (dest/'manifest.json').exists(),'Preserve a completed export'
    frame=pd.read_parquet(INPUT/'data/train_features.parquet')
    n=len(frame);assert n==70000 and frame.customer_id.is_unique
    lookup=pd.Series(np.arange(n),index=frame.customer_id)
    apps=pd.read_csv(RAW/'train_applications.csv',parse_dates=['decision_datetime'])
    apps=apps.set_index('customer_id').loc[frame.customer_id]
    assert np.array_equal(apps.application_id.to_numpy(),frame.application_id)
    cutoff=pd.Series(apps.decision_datetime.to_numpy(),index=frame.customer_id)
    audits={};files={};source_files=[INPUT/'data/train_features.parquet',RAW/'train_applications.csv']
    def save(name,**arrays):
        path=dest/(name+'.npz');np.savez_compressed(path,**arrays);files[path.name]=sha(path)
    def history(name,dates):
        source_files.append(RAW/(name+'.csv'))
        d=pd.read_csv(source_files[-1],parse_dates=dates)
        d=d[d.customer_id.isin(lookup.index)].copy()
        d['row']=d.customer_id.map(lookup).astype(np.int32)
        d['cutoff']=d.customer_id.map(cutoff)
        return d
    save('rows',target=frame.target.to_numpy(np.int8),
         covariates=frame[COV].to_numpy(float),
         declared_income=frame.declared_monthly_income.to_numpy(float),
         employment=frame.employment_type.fillna('MISSING').to_numpy(str),
         application_id=frame.application_id.to_numpy(str))
    for f in range(5):
        ixpath=INPUT/f'artifacts/fold{f}/indices.npz';ix=dict(np.load(ixpath))
        source_files.append(ixpath)
        save(f'fold{f}',**ix)
    baseline=np.full(n,np.nan);spline=baseline.copy();kumo=baseline.copy();seen=np.zeros(n,int)
    for f in range(5):
        rows=np.load(dest/f'fold{f}.npz')['validation'];seen[rows]+=1
        values=[]
        for name in ['final_oof','spline_oof']:
            path=INPUT/f'artifacts/fold{f}/{name}.parquet';source_files.append(path)
            d=pd.read_parquet(path)
            assert d.application_id.tolist()==frame.application_id.iloc[rows].tolist()
            values.append(d.prediction.to_numpy())
        baseline[rows],spline[rows]=values;kumo[rows]=(values[0]-.25*values[1])/.75
    assert np.all(seen==1) and np.all((kumo>=0)&(kumo<=1))
    save('reference',baseline=baseline,spline=spline,corrected_kumo=kumo)
    d=history('ecommerce_transactions',['transaction_datetime'])
    assert (d.transaction_datetime<d.cutoff).all()
    valid=(d.quantity>0)&(d.item_total>0);d=d[valid].copy()
    order,order_names=pd.factorize(d.order_id,sort=False)
    category,category_names=pd.factorize(d.product_category.fillna('MISSING'),sort=True)
    assert d.groupby('order_id').row.nunique().max()==1
    first=~pd.Series(order).duplicated().to_numpy()
    assert np.array_equal(order[first],np.arange(len(order_names)))
    save('prices',order=order.astype(np.int32),category=category.astype(np.int16),
         log_price=np.log(d.item_total.to_numpy(float)/d.quantity.to_numpy(float)),
         order_customer=d.row.to_numpy(np.int32)[first])
    audits['prices']=dict(items=len(d),orders=len(order_names),categories=list(map(str,category_names)),
        status_semantics='All valid basket item prices measure preference. No cancelled basket is counted as actual spending.')
    del d;gc.collect()
    d=history('bureau_observations',['report_datetime','available_datetime'])
    assert (d.report_datetime<d.cutoff).all() and (d.available_datetime<d.cutoff).all()
    d=d[d.score.notna()];providers=sorted(d.bureau_provider.unique());assert len(providers)==3
    provider=d.bureau_provider.map({name:j for j,name in enumerate(providers)})
    paired=d.pivot_table(index=['row','application_id'],columns='bureau_provider',values='score',aggfunc='mean')
    save('bureau',customer=d.row.to_numpy(np.int32),provider=provider.to_numpy(np.int8),score=d.score.to_numpy(float),
         pair_customer=paired.index.get_level_values('row').to_numpy(np.int32),pairs=paired[providers].to_numpy(float))
    audits['bureau']=dict(rows=len(d),providers=providers,paired_applications=len(paired),
        debt_semantics='Scores only; overlapping provider debt is never summed.')
    del d;gc.collect()
    d=history('account_behavior',['month'])
    mature=(d.month+pd.offsets.MonthBegin(1))<=d.cutoff
    d=d[mature & (d.income_like_credits>0)].sort_values(['row','month']).copy()
    d['log_income']=np.log(d.income_like_credits)
    g=d.groupby('row');s=g.log_income.agg(['mean','count'])
    diff=g.log_income.diff();valid=diff.notna()
    z=pd.DataFrame(dict(row=d.row[valid],diff=diff[valid],square=diff[valid]**2)).groupby('row').agg(
        diff_count=('diff','size'),diff_sum=('diff','sum'),diff_square=('square','sum'))
    all_rows=pd.DataFrame(index=np.arange(n));s=all_rows.join(s).join(z)
    save('income',mean=s['mean'].to_numpy(float),count=s['count'].fillna(0).to_numpy(int),
         diff_count=s.diff_count.fillna(0).to_numpy(int),diff_sum=s.diff_sum.fillna(0).to_numpy(float),
         diff_square=s.diff_square.fillna(0).to_numpy(float))
    audits['income']=dict(rows=len(d),complete_calendar_months_only=True,
        variance_assumption='Adjacent observed log-month income differences; stationarity diagnostic, not exact generator recovery.')
    del d;gc.collect()
    d=history('repayment_history',['due_date','last_payment_datetime'])
    assert (d.due_date<d.cutoff).all()
    assert (d.last_payment_datetime.isna()|(d.last_payment_datetime<d.cutoff)).all()
    d=d[(d.cutoff-d.due_date).dt.total_seconds().ge(86400)].copy()
    d['late']=d.max_days_past_due.ge(1).astype(int)
    s=all_rows.join(d.groupby('row').late.agg(['sum','count']))
    save('repayment',k=s['sum'].fillna(0).to_numpy(int),n=s['count'].fillna(0).to_numpy(int))
    audits['repayment']=dict(rows=len(d),maturity_days=1,threshold=1,
        point_in_time_qualification='Cutoff/maturity checked; historical max-DPD snapshot timing is unresolved. No full PIT certification.')
    manifest=dict(status='complete',created=time.time(),seconds=time.time()-start,rows=n,covariates=COV,
        raw_reference_fitting='None in export; all category, provider, income and beta priors are fitted inside each outer-training set.',
        files=files,source_files={str(p):sha(p) for p in source_files},audits=audits)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='complete',seconds=time.time()-start,bytes=sum(p.stat().st_size for p in dest.iterdir()))),flush=True)

if __name__=='__main__':run()
