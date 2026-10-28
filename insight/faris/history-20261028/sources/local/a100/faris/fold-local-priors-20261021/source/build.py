"""Rebuild inherited cohort statistics with explicit context-only membership.

Vendor snapshots are immutable. No labels enter feature estimation. Output stays
in this new study directory and never changes historical feature/model files.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,ast,json,hashlib,time,gc
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)
def load_vendor(name,work):
    path=ROOT/'vendor'/name;tree=ast.parse(path.read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,ast.If) and isinstance(node.test,ast.Compare) and '__name__' in ast.unparse(node.test):continue
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['ROOT','DESTINATION'] for t in node.targets):continue
        if isinstance(node,ast.ImportFrom) and node.module=='optimize_ks':continue
        nodes.append(node)
    ns={'__name__':'frozen_vendor','__file__':str(path),'ROOT':work,
        'OUT':work/'artifacts/optimization','DESTINATION':work/'artifacts/reverse_engineering'}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns
def check(plan):
    assert time.time()<plan['original_deadline'],'Original research deadline has expired'
    assert not (ROOT/'STOP').exists(),'Study stopped'
def build(fold):
    plan=json.loads((ROOT/'plan.json').read_text());check(plan)
    assert plan['historical_metric_recalculation_complete'] is True,'History-first metric calculation has not passed'
    assert plan['preprocessing_only'] is True and plan['model_training_admitted'] is False
    dest=ROOT/f'fold{fold}';assert not dest.exists(),'Never overwrite a completed or partial feature build'
    dest.mkdir();work=dest/'workspace';(work/'data/processed').mkdir(parents=True)
    (work/'artifacts/optimization').mkdir(parents=True)
    raw=Path(plan['raw']).resolve();(work/'data/raw').symlink_to(raw,target_is_directory=True)
    bank=Path(plan['bank']);manifest=json.loads((bank/'manifest.json').read_text())
    assert manifest['bank_id']==plan['bank_id']
    raw_hashes={name:sha(raw/(name+'.csv')) for name in ['ecommerce_transactions','bureau_observations','repayment_history','account_behavior']}
    assert all(value==manifest['raw_sha256'][name] for name,value in raw_hashes.items()),'Raw input identity differs'
    metadata=pd.read_parquet(bank/'metadata.parquet');cats=pd.read_parquet(bank/'categorical.parquet')
    lookup={n:i for i,n in enumerate(manifest['columns'])};matrix=np.load(bank/'numeric.npy',mmap_mode='r')
    view=manifest['views']['extended'];numeric=[n for n in view if n in lookup]
    base=metadata.copy();base['decision_datetime']=pd.to_datetime(base.decision_datetime)
    assert len(base)==100000 and base.application_id.is_unique and base.customer_id.is_unique
    inherited=np.array(matrix[:,[lookup[n] for n in numeric]],dtype=np.float32)
    base=pd.concat([base,pd.DataFrame(inherited,columns=numeric)],axis=1)
    for name in view:
        if name in cats:base[name]=cats[name].to_numpy()
    original=pd.read_parquet(Path(plan['input'])/'data/train_features.parquet',columns=['application_id','customer_id'])
    assert original.application_id.tolist()==base.application_id.iloc[:70000].tolist()
    assert original.customer_id.tolist()==base.customer_id.iloc[:70000].tolist()
    index=np.load(Path(plan['input'])/f'artifacts/fold{fold}/indices.npz')
    context=original.iloc[index['context']];validation=original.iloc[index['validation']]
    assert len(context)==50000 and len(validation)==14000
    assert not set(context.application_id)&set(validation.application_id)
    fit_ids=set(context.customer_id);base['split']=np.where(base.customer_id.isin(fit_ids),'train','test')
    assert int((base.split=='train').sum())==50000
    # Old builders require a target column for equality assertions. They receive
    # a constant placeholder, never the real training or evaluation targets.
    base['target']=0
    base.to_parquet(work/'data/processed/application_features_extended.parquet',index=False)
    bureau=pd.read_csv(raw/'bureau_observations.csv',parse_dates=['report_datetime','available_datetime'])
    cutoff=base.set_index('customer_id').decision_datetime
    decision=bureau.customer_id.map(cutoff)
    assert (bureau.report_datetime<decision).all() and (bureau.available_datetime<decision).all()
    latest=bureau.sort_values(['report_datetime','available_datetime','bureau_observation_id']).groupby(['customer_id','bureau_provider']).tail(1)
    latest.to_parquet(work/'data/processed/bureau_latest.parquet',index=False)
    del bureau,latest
    del matrix,inherited;gc.collect()
    sources={};start=time.time()
    for name in ['build_features_v3.py','build_features_v4.py','build_features_v5.py']:
        check(plan);sources[name]=sha(ROOT/'vendor'/name)
        load_vendor(name,work)['main']();gc.collect()
    full=pd.read_parquet(work/'data/processed/application_features_v5.parquet').set_index('customer_id')
    # Only statistically specified latent estimators are invoked. Diagnostic
    # rule reconstruction and RNG probes are excluded from this build.
    ns=load_vendor('reverse_synthetic_patterns.py',work);sources['reverse_synthetic_patterns.py']=sha(ROOT/'vendor/reverse_synthetic_patterns.py')
    blocks=[];audits={}
    for label,fn in [('price','fit_price_latent'),('bureau','fit_bureau_latent'),('repayment','fit_repayment_latent')]:
        check(plan);values=ns[fn](full);blocks.append(values[0]);audits[label]=values[1];gc.collect()
    # Account means/gaps are row-local. Its employment-wide noise variance must
    # additionally be learned from context histories only.
    account=pd.read_csv(raw/'account_behavior.csv',parse_dates=['month'])
    account=account.merge(full[['decision_datetime']],left_on='customer_id',right_index=True,validate='many_to_one')
    assert ((account.month+pd.offsets.MonthBegin(1))<=account.decision_datetime).all()
    account['li']=np.log(account.income_like_credits.where(account.income_like_credits>0))
    account['lo']=np.log(account.outflows.where(account.outflows>0))
    account=account.sort_values(['customer_id','month'])
    account['employment_type']=account.customer_id.map(full.employment_type)
    account['income_difference']=account.groupby('customer_id').li.diff()
    account_fit=account[account.customer_id.isin(fit_ids)]
    variance=account_fit.groupby('employment_type').income_difference.var(ddof=0)/2
    summary=account.groupby('customer_id')[['li','lo']].mean().reindex(full.index)
    count=account.groupby('customer_id').li.count().reindex(full.index)
    logsd=account.groupby('customer_id').li.std().reindex(full.index)
    measurement_variance=full.employment_type.map(variance)/count.where(count>0)
    blocks.append(pd.DataFrame(dict(re_income_geometric_mean=np.exp(summary.li),
        re_income_log_mean=summary.li,re_income_log_sd=logsd,
        re_income_log_measurement_sd=np.sqrt(measurement_variance),
        re_declared_income_log_gap=np.log(full.declared_monthly_income)-summary.li,
        re_outflow_income_geometric_ratio=np.exp(summary.lo-summary.li)),index=full.index))
    audits['account_noise_context_variance']=variance.to_dict()
    rebuilt=full.join(pd.concat(blocks,axis=1)).replace([np.inf,-np.inf],np.nan)
    lock=json.loads((Path(plan['input'])/'BASELINE_LOCK.json').read_text())
    required=list(dict.fromkeys(lock['Kumo']['features']+lock['correction']['features']+lock['original_spline']['features']))
    assert set(required)<=set(rebuilt.columns),sorted(set(required)-set(rebuilt.columns))
    assert not {'target','customer_id','application_id'}&set(required)
    assert rebuilt.target.eq(0).all()
    table=rebuilt.reset_index()[['application_id','customer_id']+required]
    table.to_parquet(dest/'features.parquet',index=False)
    dump(dest/'prior_audit.json',audits)
    dump(dest/'complete.json',dict(fold=fold,context_rows=50000,validation_rows=14000,
        fit_ids_sha256=hashlib.sha256('\n'.join(context.application_id).encode()).hexdigest(),
        feature_table_sha256=sha(dest/'features.parquet'),source_sha256=sources,
        raw_sha256=raw_hashes,
        no_target_labels_available_to_feature_estimators=True,training_only_cohort_statistics=True,
        base_precision='Inherited float32 row-local extended feature bank; shared by all new conditions.',
        original_snapshot_certification_unchanged=False,seconds=time.time()-start))
    print(json.dumps(dict(stage='fold_features_complete',fold=fold,seconds=time.time()-start)),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(5),required=True)
    build(parser.parse_args().fold)
