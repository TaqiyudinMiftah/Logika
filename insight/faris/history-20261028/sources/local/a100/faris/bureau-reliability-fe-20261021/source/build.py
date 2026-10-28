"""Persistent provider effects versus repeated-report noise; labels unavailable."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'

def fit(summary,fit_ids):
 train=summary[summary.customer_id.isin(fit_ids)].copy();names=sorted(train.bureau_provider.unique());assert len(names)==3
 # Within-customer/provider residual variance; real temporal credit changes can
 # also contribute. Treat this as a measurement model, not recovered truth.
 sigma=train.groupby('bureau_provider').sse.sum()/train.groupby('bureau_provider').df.sum()
 means=train.pivot(index='customer_id',columns='bureau_provider',values='score')
 counts=train.pivot(index='customer_id',columns='bureau_provider',values='n')
 pairs=[(names[0],names[1]),(names[0],names[2]),(names[1],names[2])]
 rows=[];deltas=[];variances=[];diagnostics={}
 for pa,pb in pairs:
  dif=(means[pa]-means[pb]).dropna();ix=dif.index
  noise=float((sigma[pa]/counts.loc[ix,pa]+sigma[pb]/counts.loc[ix,pb]).mean())
  residual=float(dif.var(ddof=0)-noise)
  rows.append([int(p==pa)-int(p==pb)for p in names]);deltas.append(float(dif.mean()));variances.append(residual)
  diagnostics[pa+'-'+pb]=dict(customers=len(ix),difference_variance=float(dif.var(ddof=0)),estimated_mean_measurement_variance=noise,excess_between_provider_variance=residual)
 # Symmetric least-squares origin, fixed sum(offset)=0.
 offsets=np.linalg.lstsq(np.vstack([rows,np.ones(3)]),np.r_[deltas,0.],rcond=None)[0]
 tau_unconstrained=np.array([(variances[0]+variances[1]-variances[2])/2,(variances[0]+variances[2]-variances[1])/2,(variances[1]+variances[2]-variances[0])/2])
 tau=np.maximum(tau_unconstrained,0.)
 params=dict(names=names,offset=dict(zip(names,offsets.tolist())),sigma=sigma.to_dict(),tau=dict(zip(names,tau.tolist())),unconstrained_tau=dict(zip(names,tau_unconstrained.tolist())),pair_diagnostics=diagnostics)
 m=aggregate(summary,params);t=m.loc[m.index.isin(fit_ids)]
 mu=float(t.mean_score.mean());variance=max(float(t.mean_score.var(ddof=0)-t.measurement_variance.mean()),1e-8)
 params.update(prior_mean=mu,prior_variance=variance,fit_customers=len(fit_ids));return params

def aggregate(summary,params):
 d=summary.copy();d['variance']=d.bureau_provider.map(params['tau'])+d.bureau_provider.map(params['sigma'])/d.n
 d['precision']=1/d.variance.clip(lower=1e-8);d['weighted']=(d.score-d.bureau_provider.map(params['offset']))*d.precision
 g=d.groupby('customer_id')[['precision','weighted']].sum()
 return pd.DataFrame(dict(mean_score=g.weighted/g.precision,measurement_variance=1/g.precision))

def transform(summary,params,customers):
 m=aggregate(summary,params).reindex(customers);precision=1/m.measurement_variance
 variance=1/(precision.fillna(0)+1/params['prior_variance'])
 posterior=variance*((m.mean_score*precision).fillna(0)+params['prior_mean']/params['prior_variance'])
 return pd.DataFrame(dict(fe_bureau_correlated_posterior=posterior,fe_bureau_correlated_posterior_sd=np.sqrt(variance)),index=customers)

def main():
 start=time.time();assert not(R/'BUILD_COMPLETE.json').exists();assert time.time()<1792585871.823748-10800
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True)
 assert len(apps)==100000 and apps.customer_id.is_unique;apps.decision_datetime=pd.to_datetime(apps.decision_datetime)
 raw=RAW/'bureau_observations.csv';d=pd.read_csv(raw,parse_dates=['report_datetime','available_datetime'],usecols=['customer_id','bureau_provider','report_datetime','available_datetime','score'])
 d=d.merge(apps,on='customer_id',validate='many_to_one');assert (d.available_datetime<d.decision_datetime).all()and(d.report_datetime<d.decision_datetime).all()
 d=d.sort_values('available_datetime').drop_duplicates(['customer_id','bureau_provider','report_datetime'],keep='last');d=d[d.score.notna()].copy()
 group=['customer_id','bureau_provider'];d['residual']=d.score-d.groupby(group).score.transform('mean');d['ss']=d.residual**2
 summary=d.groupby(group).agg(score=('score','mean'),n=('score','size'),sse=('ss','sum')).reset_index();summary['df']=summary.n-1
 summary.to_parquet(R/'provider_summary.parquet',index=False)
 rows=[]
 original_ids=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);assert apps.application_id.iloc[:70000].tolist()==original_ids.application_id.tolist()
 for f in range(5):
  assert time.time()<1792585871.823748-10800
  ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');fit_ids=set(original_ids.customer_id.iloc[ix['context']]);params=fit(summary,fit_ids)
  result=transform(summary,params,pd.Index(apps.customer_id,name='customer_id')).reset_index();result.insert(0,'application_id',apps.application_id.to_numpy())
  out=R/f'fold{f}';out.mkdir();result.to_parquet(out/'features.parquet',index=False)
  b.dump(params,out/'transformer.json');b.dump(dict(fold=f,fit_rows=50000,fit_ids_sha256=b.ids_hash(original_ids,ix['context']),query_covariates_fit_shared_parameters=False,target_labels_opened=False,features_hash=b.sha(out/'features.parquet'),transformer_hash=b.sha(out/'transformer.json')),out/'boundary.json')
  # Provider contrasts from early/late halves cancel common borrower score;
  # their covariance tests persistent provider differences without default labels.
  train=d[d.customer_id.isin(fit_ids)].sort_values(['customer_id','bureau_provider','report_datetime']).copy()
  gg=train.groupby(group);train['position']=gg.cumcount();train['count']=gg.score.transform('size');train=train[train['count']>=2].copy();train['half']=(train.position>=train['count']//2).astype(int)
  h=train.pivot_table(index=['customer_id','half'],columns='bureau_provider',values='score',aggfunc='mean');names=params['names'];repeat={}
  for pa,pb in [(names[0],names[1]),(names[0],names[2]),(names[1],names[2])]:
   pair=(h[pa]-h[pb]).unstack('half').dropna();repeat[pa+'-'+pb]=dict(customers=len(pair),early_late_contrast_covariance=float(np.cov(pair[0],pair[1],ddof=0)[0,1]),early_late_contrast_correlation=float(pair[0].corr(pair[1])))
  rows.append(dict(fold=f,params=params,repeat_diagnostic=repeat));print(json.dumps(dict(stage='feature_fold_complete',fold=f,tau=params['tau'],repeat=repeat)),flush=True)
 b.dump(dict(status='complete',rows=100000,feature_count=2,folds=rows,target_labels_opened=False,source_bureau_sha256=b.sha(raw),provider_summary_sha256=b.sha(R/'provider_summary.parquet'),source_sha256=b.sha(Path(__file__)),seconds=time.time()-start,qualification='Random-effects measurement hypothesis; genuine temporal change, correlated errors across providers and informative reporting can invalidate assumptions. No claim of latent truth. No model training or label metric used.'),R/'BUILD_COMPLETE.json')
if __name__=='__main__':main()
