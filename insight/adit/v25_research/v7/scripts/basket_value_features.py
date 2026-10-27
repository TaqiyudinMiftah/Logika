"""Bounded-memory item-value composition, strictly before each application."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7';RAW=ROOT.parents[1]/'data/raw'
OUT=V/'light_fe';CATEGORIES=['groceries','fashion','electronics','home','health','education','personal_care']


def memory():
    text=Path('/proc/self/status').read_text();d={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in text.splitlines() if ':' in x}
    return dict(RSS_MiB=int(d['VmRSS'].split()[0])/1024,peak_RSS_MiB=int(d['VmHWM'].split()[0])/1024,cgroup_available_GiB=(int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text()))/2**30)


def state(status,**extra):
    (OUT/'reports').mkdir(parents=True,exist_ok=True)
    report=dict(status=status,pid=os.getpid(),UTC=datetime.now(timezone.utc).isoformat(),memory=memory(),**extra)
    target=OUT/'reports/BASKET_VALUE_BUILD.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(target)


def stats(spend):
    total=spend.sum(1);p=np.divide(spend,total[:,None],out=np.zeros_like(spend),where=total[:,None]>0)
    entropy=-np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)
    entropy[total<=0]=np.nan
    return total,entropy,np.where(total>0,np.sum(p*p,axis=1),np.nan),(spend>0).sum(1)


def main():
    started=time.time();state('LOADING_APPLICATION_CUTOFFS')
    cols=['application_id','customer_id','decision_datetime']
    a=pd.concat([pd.read_csv(RAW/'train_applications.csv',usecols=cols),pd.read_csv(RAW/'test_applications.csv',usecols=cols)],ignore_index=True)
    assert a.application_id.is_unique and a.customer_id.is_unique and len(a)==100000
    cutoff=pd.to_datetime(a.decision_datetime).to_numpy(dtype='datetime64[ns]').astype(np.int64)
    lookup=pd.Index(a.customer_id);n=len(a);spend=np.zeros((2,n,7),dtype=np.float64);items=np.zeros((2,n),dtype=np.int32)
    seen=eligible=cancelled=unknown=0
    use=['customer_id','transaction_datetime','product_category','item_total','order_status']
    source=RAW/'ecommerce_transactions.csv';schema=pd.read_csv(source,nrows=0).columns.tolist();assert all(c in schema for c in use)
    for number,d in enumerate(pd.read_csv(source,usecols=use,chunksize=50000)):
        seen+=len(d);indices=lookup.get_indexer(d.customer_id);unknown+=int((indices<0).sum())
        completed=d.order_status.eq('completed').to_numpy();cancelled+=int(d.order_status.eq('cancelled').sum())
        selected=(indices>=0)&completed
        if selected.any():
            x=d.loc[selected];ix=indices[selected];event=pd.to_datetime(x.transaction_datetime).to_numpy(dtype='datetime64[ns]').astype(np.int64)
            age=(cutoff[ix]-event)/(86400*1e9);cat=pd.Categorical(x.product_category,categories=CATEGORIES).codes
            amounts=x.item_total.to_numpy(np.float64);assert np.all(cat>=0) and np.isfinite(amounts).all() and np.all(amounts>=0)
            eligible+=int(((age>0)&(age<=90)).sum())
            for w,days in enumerate([30,90]):
                m=(age>0)&(age<=days)
                np.add.at(spend[w],(ix[m],cat[m]),amounts[m]);np.add.at(items[w],ix[m],1)
        if number%10==0:
            state('STREAMING_ITEM_EVENTS',chunks=number+1,rows_read=seen,eligible90d_item_rows=eligible,cancelled_rows_excluded=cancelled)
            print('BASKET_VALUE_CHUNK',number+1,seen,memory(),flush=True)
    result=pd.DataFrame({'application_id':a.application_id})
    for w,days in enumerate([30,90]):
        total,entropy,hhi,ncat=stats(spend[w]);result[f'light_basket_value_entropy_{days}d']=entropy
        result[f'light_basket_value_hhi_{days}d']=hhi;result[f'light_basket_observed_categories_{days}d']=ncat
        result[f'light_basket_completed_item_rows_{days}d']=items[w];result[f'light_basket_completed_spend_{days}d']=total
    prior=spend[1]-spend[0];assert np.min(prior)>=-1e-5
    _,earlier,_,_=stats(np.maximum(prior,0))
    result['light_basket_value_entropy_change_30_vs_prior60']=result.light_basket_value_entropy_30d-earlier
    (OUT/'data').mkdir(exist_ok=True);target=OUT/'data/basket_value_features.parquet';result.to_parquet(target,index=False)
    record=dict(rows=len(result),rows_read=seen,eligible90d_item_rows=eligible,cancelled_item_rows_excluded=cancelled,unknown_customer_item_rows=unknown,feature_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_file_bytes=source.stat().st_size,source_file_mtime_ns=source.stat().st_mtime_ns,source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dictionary_sha256=hashlib.sha256((RAW/'DATA_DICTIONARY.md').read_bytes()).hexdigest(),strict_cutoff=True,order_total_not_summed=True,item_total_used_for_category_value=True,no_targets_used=True,opaque_ids_only_join_keys=True,no_global_duplicate_claim='Rows preserved at item grain; existing full duplicate audit not repeated here',elapsed_seconds=time.time()-started)
    state('COMPLETE',**record);print('BASKET_VALUE_READY',json.dumps(record),flush=True)


if __name__=='__main__':main()
