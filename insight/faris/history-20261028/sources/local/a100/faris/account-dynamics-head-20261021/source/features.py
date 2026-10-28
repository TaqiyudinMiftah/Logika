"""Unsupervised history dynamics versus within-customer shuffled time order."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib,numpy as np,pandas as pd
from scipy.stats import t
R=Path(__file__).resolve().parents[1];raw=R.parent/'exploration-12h-20261017/raw';assert time.time()<1792585871.823748-10800
cols=['application_id','customer_id','decision_datetime'];apps=pd.concat([pd.read_csv(raw/(name+'_applications.csv'),usecols=cols)for name in ['train','test']],ignore_index=True);assert len(apps)==100000 and apps.customer_id.is_unique
apps.decision_datetime=pd.to_datetime(apps.decision_datetime)
a=pd.read_csv(raw/'account_behavior.csv',usecols=['customer_id','month','income_like_credits','outflows','low_balance_days']);a.month=pd.to_datetime(a.month);a=a.merge(apps,on='customer_id',validate='many_to_one');assert (a.month+pd.offsets.MonthBegin(1)<=a.decision_datetime).all();assert not a.duplicated(['customer_id','month']).any()
a=a.sort_values(['customer_id','month']).reset_index(drop=True);codes,ids=pd.factorize(a.customer_id,sort=False);g=len(ids);n=np.bincount(codes);tm=(a.month.dt.year*12+a.month.dt.month).to_numpy(float);center=tm-np.bincount(codes,weights=tm)[codes]/n[codes];sxx=np.bincount(codes,weights=center*center);good=(n>=6)&(sxx>0)
variables=dict(log_income=np.log(a.income_like_credits.where(a.income_like_credits>0)).to_numpy(),log_outflow=np.log(a.outflows.where(a.outflows>0)).to_numpy(),low_balance_fraction=(a.low_balance_days/a.month.dt.days_in_month).to_numpy())
# Whole observed monthly vectors stay together in the null; only their order changes.
rng=np.random.default_rng(20261023);permutation=np.lexsort((rng.random(len(a)),codes));assert np.array_equal(codes[permutation],codes)
summary={};features={};null_features={};previous=np.r_[False,(codes[1:]==codes[:-1])&(tm[1:]-tm[:-1]==1)]
for name,values in variables.items():
 finite=np.isfinite(values);assert finite.all(),name
 def estimate(y):
  mean=np.bincount(codes,weights=y)/n;yc=y-mean[codes];slope=np.divide(np.bincount(codes,weights=center*yc),sxx,out=np.zeros(g),where=sxx>0);res=yc-slope[codes]*center;rss=np.bincount(codes,weights=res*res);se=np.sqrt(np.divide(rss,np.maximum(n-2,1)*np.maximum(sxx,1e-12)));stat=np.divide(slope,se,out=np.zeros(g),where=se>1e-12);threshold=t.ppf(.975,np.maximum(n-2,1));lag=yc[previous];prev=yc[np.flatnonzero(previous)-1];corr=float(lag@prev/np.sqrt((lag@lag)*(prev@prev)))
  return dict(significant_slope_fraction=float(np.mean(abs(stat[good])>threshold[good])),slope_quantiles=np.quantile(slope[good],[.01,.1,.5,.9,.99]).tolist(),pooled_demeaned_contiguous_lag_correlation=corr),slope,se
 real,slope,se=estimate(values);null,ns,nse=estimate(values[permutation]);null_features['dyn_'+name+'_slope']=ns;null_features['dyn_'+name+'_slope_SE']=nse;summary[name]=dict(observed=real,permuted_time_null=null,significant_slope_fraction_difference=real['significant_slope_fraction']-null['significant_slope_fraction'],lag_correlation_difference=real['pooled_demeaned_contiguous_lag_correlation']-null['pooled_demeaned_contiguous_lag_correlation']);features['dyn_'+name+'_slope']=slope;features['dyn_'+name+'_slope_SE']=se
f=pd.DataFrame(features,index=ids);f.index.name='customer_id';f['dyn_account_n']=n;f.reset_index().to_parquet(R/'row_local_features.parquet',index=False);nf=pd.DataFrame(null_features,index=ids);nf.index.name='customer_id';nf.reset_index().to_parquet(R/'null_row_local_features.parquet',index=False)
result=dict(status='complete',target_labels_opened=False,customers=g,rows=len(a),observed_month_count_quantiles=np.quantile(n,[0,.1,.5,.9,1]).tolist(),minimum_months_for_slope_diagnostic=6,source_account_sha256=hashlib.sha256((raw/'account_behavior.csv').read_bytes()).hexdigest(),summary=summary,null='One seeded within-customer multivariate month-order permutation; no target or customer exchange. Preserves customer distribution and coverage; destroys time ordering.',qualification='Diagnostic only, not causal or performance evidence. Trend t-test uses iid errors, so permutation comparison is descriptive. Features are per-customer OLS, no shared fit; no new model training admitted.')
(R/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
