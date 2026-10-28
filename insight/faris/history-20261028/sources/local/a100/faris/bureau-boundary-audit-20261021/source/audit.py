"""Context-only score edge/rounding audit. No applicant default labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np,pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def main():
 assert not(R/'RESULT.json').exists()and time.time()<1792585871.823748-10800;start=time.time()
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);apps=pd.read_csv(RAW/'train_applications.csv',usecols=['application_id','customer_id','decision_datetime']);apps=apps.set_index('application_id').loc[original.application_id].reset_index();apps.decision_datetime=pd.to_datetime(apps.decision_datetime);ix=np.load(INPUT/'artifacts/fold0/indices.npz');context=apps.iloc[ix['context']];assert len(context)==50000
 raw=RAW/'bureau_observations.csv';d=pd.read_csv(raw,usecols=['customer_id','application_id','bureau_provider','report_datetime','available_datetime','score'],parse_dates=['report_datetime','available_datetime']);d=d.rename(columns={'application_id':'source_application_id'}).merge(context,on='customer_id',validate='many_to_one');assert d.report_datetime.lt(d.decision_datetime).all()and d.available_datetime.lt(d.decision_datetime).all();before=len(d);d=d.sort_values('available_datetime').drop_duplicates(['customer_id','bureau_provider','report_datetime'],keep='last');finite=d[np.isfinite(d.score)].copy();assert len(finite)>0
 rows=[]
 for provider,x in finite.groupby('bureau_provider'):
  low=float(x.score.min());high=float(x.score.max());edge=x.score.eq(low)|x.score.eq(high);customers=x.groupby('customer_id').score.agg(['size','min','max'])
  rows.append(dict(provider=provider,scored_rows=len(x),customers=int(x.customer_id.nunique()),minimum=low,maximum=high,at_min_count=int(x.score.eq(low).sum()),at_max_count=int(x.score.eq(high).sum()),edge_fraction=float(edge.mean()),customers_touching_edge=int(x.loc[edge,'customer_id'].nunique()),lower_neighbor_counts={str(low+j):int(x.score.eq(low+j).sum())for j in range(4)},upper_neighbor_counts={str(high-j):int(x.score.eq(high-j).sum())for j in range(4)},score_quantiles=x.score.quantile([0,.001,.01,.1,.5,.9,.99,.999,1]).tolist(),integer_fraction=float(x.score.mod(1).eq(0).mean())))
  finite.loc[x.index,'at_observed_edge']=edge.to_numpy()
 overall=dict(source_rows=before,unique_reports=len(d),scored_rows=len(finite),score_missing_rows=int(d.score.isna().sum()),customers_with_any_observed_edge=int(finite.loc[finite.at_observed_edge.astype(bool),'customer_id'].nunique()),scored_customers=int(finite.customer_id.nunique()),edge_rows=int(finite.at_observed_edge.sum()))
 prior=json.loads((A/'fold-local-priors-20261021/fold0/prior_audit.json').read_text())['bureau'];known={k:v for k,v in prior.items()if any(t in k for t in ['provider_noise','provider_offsets','latent_score_prior'])}
 pd.DataFrame(rows).to_csv(R/'provider_edges.csv',index=False)
 b.dump(dict(status='complete',default_labels_opened=False,context_rows=50000,fit_ids_sha256=b.ids_hash(original,ix['context']),summary=overall,providers=rows,existing_gaussian_estimator_parameters=known,source_raw_sha256=b.sha(raw),source_sha256=b.sha(Path(__file__)),seconds=time.time()-start,classification_training_admitted=False,qualification='Observedmin/maxareempiricaledges,notprovenclippingbounds. Rareedgesdonotproveabsenceofdistributionalbias. No default-scoring result or latent data-generating process is inferred.'),R/'RESULT.json');print(json.dumps(dict(summary=overall,providers=rows)),flush=True)
if __name__=='__main__':main()
