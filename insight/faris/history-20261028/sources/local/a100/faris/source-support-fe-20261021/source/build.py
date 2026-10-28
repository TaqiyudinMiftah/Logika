import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time,json
import numpy as np,pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw'
EXTRA=['support_full_account_months','support_distinct_scored_bureau_reports']
def main():
 assert not(R/'BUILD_COMPLETE.json').exists()and time.time()<1792585871.823748-10800;start=time.time()
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True);assert apps.customer_id.is_unique and len(apps)==100000;apps.decision_datetime=pd.to_datetime(apps.decision_datetime);apps['row']=np.arange(len(apps));out=apps[['application_id','customer_id']].copy()
 ac=pd.read_csv(RAW/'account_behavior.csv',usecols=['customer_id','month'],parse_dates=['month']).merge(apps,on='customer_id',validate='many_to_one');assert not ac.duplicated(['customer_id','month']).any();ac=ac[ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime]
 counts=ac.groupby('customer_id').size().reindex(apps.customer_id,fill_value=0).to_numpy(np.int64);independent=np.bincount(ac.row.to_numpy(),minlength=len(apps));assert np.array_equal(counts,independent);out[EXTRA[0]]=counts
 bu=pd.read_csv(RAW/'bureau_observations.csv',usecols=['customer_id','bureau_provider','report_datetime','available_datetime','score'],parse_dates=['report_datetime','available_datetime']).merge(apps,on='customer_id',validate='many_to_one');bu=bu[bu.report_datetime.lt(bu.decision_datetime)&bu.available_datetime.lt(bu.decision_datetime)].copy();original=len(bu);bu=bu.sort_values('available_datetime').drop_duplicates(['customer_id','bureau_provider','report_datetime'],keep='last');dedup=original-len(bu);bu=bu[np.isfinite(bu.score)]
 counts=bu.groupby('customer_id').size().reindex(apps.customer_id,fill_value=0).to_numpy(np.int64);independent=np.bincount(bu.row.to_numpy(),minlength=len(apps));assert np.array_equal(counts,independent);out[EXTRA[1]]=counts
 assert not{'target','split'}&set(out);out.to_parquet(R/'features.parquet',index=False)
 b.dump(dict(status='complete',default_labels_opened=False,shared_parameters_fitted=False,rows=len(out),features=EXTRA,source_hashes={n:b.sha(RAW/n)for n in ['account_behavior.csv','bureau_observations.csv']},feature_hash=b.sha(R/'features.parquet'),independent_NumPy_counts_exact=True,bureau_duplicate_versions_removed=dedup,nonconstant_counts={n:int(out[n].nunique())for n in EXTRA},quantiles={n:np.quantile(out[n],[0,.1,.5,.9,1]).tolist()for n in EXTRA},source_sha256=b.sha(Path(__file__)),seconds=time.time()-start),R/'BUILD_COMPLETE.json');print(json.dumps(dict(status='complete',seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
