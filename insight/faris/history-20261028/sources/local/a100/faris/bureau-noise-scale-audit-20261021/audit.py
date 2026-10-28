"""Target-free check of the inherited bureau measurement unit, not a score search."""
from pathlib import Path
import json,hashlib,time
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent;A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 start=time.time();assert not(R/'RESULT.json').exists()
 ids=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id']);ix=np.load(INPUT/'artifacts/fold0/indices.npz');context=set(ids.application_id.iloc[ix['context']]);apps=pd.read_csv(RAW/'train_applications.csv',usecols=['application_id','customer_id','decision_datetime'],parse_dates=['decision_datetime']);apps=apps[apps.application_id.isin(context)];assert len(apps)==50000 and apps.customer_id.is_unique
 customer=set(apps.customer_id);cutoff=apps.set_index('customer_id').decision_datetime
 d=pd.read_csv(RAW/'bureau_observations.csv',parse_dates=['report_datetime','available_datetime']);d=d[d.customer_id.isin(customer)].copy();assert(d.report_datetime<d.customer_id.map(cutoff)).all()and(d.available_datetime<d.customer_id.map(cutoff)).all();assert not d.duplicated(['customer_id','bureau_provider','report_datetime']).any()
 finite=d[d.score.notna()].copy();group=finite.groupby(['customer_id','application_id','bureau_provider']).agg(reports=('score','size'),score_mean=('score','mean'),first_report=('report_datetime','min'),last_report=('report_datetime','max')).reset_index();group['span_days']=(group.last_report-group.first_report).dt.total_seconds()/86400
 summary={}
 for provider,g in group.groupby('bureau_provider'):
  summary[str(provider)]=dict(groups=len(g),reports=int(g.reports.sum()),report_count_histogram={str(k):int(v)for k,v in g.reports.value_counts().sort_index().items()},multireport_groups=int((g.reports>1).sum()),span_days_quantiles=g.span_days.quantile([0,.5,.9,.99,1]).tolist())
 paired=group.pivot(index=['customer_id','application_id'],columns='bureau_provider',values='score_mean');counts=group.pivot(index=['customer_id','application_id'],columns='bureau_provider',values='reports');pairs=[];names=sorted(paired.columns)
 for i,a in enumerate(names):
  for b in names[i+1:]:
   mask=paired[a].notna()&paired[b].notna();na=counts.loc[mask,a];nb=counts.loc[mask,b]
   pairs.append(dict(providers=[str(a),str(b)],paired_groups=int(mask.sum()),any_averaged=int(((na>1)|(nb>1)).sum()),score_difference_variance=float((paired.loc[mask,a]-paired.loc[mask,b]).var(ddof=0)),mean_inverse_counts=[float((1/na).mean()),float((1/nb).mean())]))
 outcome='NO_AVERAGING_SCALE_MISMATCH'if not(group.reports>1).any()else'AVERAGING_SCALE_REQUIRES_FURTHER_TARGET_FREE_CHECK'
 result=dict(status='complete',diagnostic=outcome,context_customers=50000,default_labels_opened=False,classification_fit_admitted=False,provider_groups=summary,paired_noise_units=pairs,source_hashes={str(p):sha(p)for p in[RAW/'bureau_observations.csv',RAW/'train_applications.csv',INPUT/'artifacts/fold0/indices.npz',Path(__file__)]},seconds=time.time()-start,qualification='Checks whether inherited paired means average multiple finite reports. Does not establish independent/unbiased/time-stationary provider errors or snapshot semantics.')
 (R/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status='complete',diagnostic=outcome,provider_groups=summary,seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
