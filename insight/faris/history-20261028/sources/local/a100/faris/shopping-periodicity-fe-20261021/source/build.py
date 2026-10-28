import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,ast
import numpy as np,pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw'
NAMES=['power_7d','power_14d','power_30d','periodicity_strength','dominant_cycle_days']
EXTRA=['beh_spectral_'+channel+'_'+name for channel in ['spend','orders']for name in NAMES]+['beh_spectral_window_supported']
def spectral_batch(y):
 y=np.asarray(y,float);t=np.arange(180,dtype=float);t-=t.mean();d=y-y.mean(axis=1)[:,None]-(y@t/(t@t))[:,None]*t
 power=abs(np.fft.rfft(d*np.hanning(180),axis=1))**2;power[:,0]=0;total=power.sum(axis=1);freq=np.fft.rfftfreq(180);period=np.divide(1,freq,out=np.full_like(freq,np.inf),where=freq>0)
 out=np.full((len(y),5),np.nan);good=total>1e-10
 for j,(lo,hi)in enumerate([(6,8),(12,16),(25,35)]):out[good,j]=power[good][:,(period>=lo)&(period<=hi)].sum(axis=1)/total[good]
 eligible=np.flatnonzero((period>=3)&(period<=90));peak=eligible[np.argmax(power[:,eligible],axis=1)];out[good,3]=power[np.flatnonzero(good),peak[good]]/total[good];out[good,4]=period[peak[good]];return out
def main():
 assert not(R/'BUILD_COMPLETE.json').exists()and time.time()<1792585871.823748-10800;start=time.time()
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True);assert len(apps)==100000 and apps.customer_id.is_unique;apps.decision_datetime=pd.to_datetime(apps.decision_datetime);apps['row']=np.arange(len(apps))
 raw=RAW/'ecommerce_transactions.csv';d=pd.read_csv(raw,usecols=['customer_id','order_id','transaction_datetime','order_total','order_status'],parse_dates=['transaction_datetime']).merge(apps,on='customer_id',validate='many_to_one');assert d.transaction_datetime.lt(d.decision_datetime).all();assert d.groupby('order_id').customer_id.nunique().max()==1 and d.groupby('order_id').order_total.nunique().max()==1 and d.groupby('order_id').order_status.nunique().max()==1
 orders=d.drop_duplicates('order_id').copy();begin=apps.decision_datetime.dt.normalize()-pd.Timedelta(days=180);first=orders.groupby('row').transaction_datetime.min().reindex(apps.row);support=first.to_numpy()<=begin.to_numpy();orders['bin']=(orders.transaction_datetime.dt.normalize()-orders.row.map(begin)).dt.days;q=orders[orders.order_status.eq('completed')&orders['bin'].between(0,179)]
 spend=np.zeros((len(apps),180),np.float32);counts=np.zeros_like(spend);idx=(q.row.to_numpy(int),q['bin'].to_numpy(int));np.add.at(spend,idx,q.order_total.to_numpy(float));np.add.at(counts,idx,1.);enough=support&(counts.sum(axis=1)>=8);actual=np.full((len(apps),10),np.nan);null=np.full_like(actual,np.nan);rng=np.random.default_rng(20261127);rows=np.flatnonzero(enough)
 for startrow in range(0,len(rows),1024):
  rr=rows[startrow:startrow+1024];actual[rr,:5]=spectral_batch(spend[rr]);actual[rr,5:]=spectral_batch(counts[rr]);permutation=np.argsort(rng.random((len(rr),180)),axis=1);ns=np.take_along_axis(spend[rr],permutation,axis=1);nc=np.take_along_axis(counts[rr],permutation,axis=1)
  assert np.allclose(ns.sum(axis=1,dtype=np.float64),spend[rr].sum(axis=1,dtype=np.float64),rtol=0,atol=0)and np.array_equal(nc.sum(axis=1),counts[rr].sum(axis=1));null[rr,:5]=spectral_batch(ns);null[rr,5:]=spectral_batch(nc)
 # Compare independently batched implementation with the inspected source's
 # pure spectral function only; do not execute the source module or its imports.
 source=R/'vendor/behavior_features.py';tree=ast.parse(source.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='spectral');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
 sample=rows[np.random.default_rng(20261128).choice(len(rows),min(128,len(rows)),replace=False)]
 for rr in sample:
  assert np.allclose(actual[rr,:5],ns['spectral'](spend[rr]),rtol=0,atol=1e-10,equal_nan=True)and np.allclose(actual[rr,5:],ns['spectral'](counts[rr]),rtol=0,atol=1e-10,equal_nan=True)
 wave=10+np.sin(2*np.pi*np.arange(180)/7);known=spectral_batch(wave[None])[0];assert 6<=known[-1]<=8 and known[0]>.8;assert np.isnan(spectral_batch(np.ones((1,180)))).all()
 frames={}
 for kind,values in [('true',actual),('shuffled',null)]:
  f=apps[['application_id','customer_id']].copy()
  for j,name in enumerate(EXTRA[:-1]):f[name]=values[:,j]
  f[EXTRA[-1]]=support.astype(float);f.to_parquet(R/(kind+'_features.parquet'),index=False);frames[kind]=f
 reference=pd.read_parquet(R/'evidence/teammate_spectral.parquet').set_index('application_id').loc[apps.application_id];errors={}
 for c in EXTRA:
  observed=frames['true'][c].to_numpy(float);expected=reference[c].to_numpy(float);assert np.array_equal(np.isnan(observed),np.isnan(expected)),c;error=float(np.nanmax(abs(observed-expected)));assert error<=1e-6,(c,error)
  if'dominant_cycle'in c or'window_supported'in c:assert np.array_equal(observed[np.isfinite(observed)],expected[np.isfinite(expected)]),c
  errors[c]=error
 b.dump(dict(status='complete',rows=100000,features=EXTRA,default_labels_opened=False,shared_parameters_fitted=False,eligible_spectral_customers=int(enough.sum()),window_supported_customers=int(support.sum()),vendor_source_sha256=b.sha(source),source_raw_sha256=b.sha(raw),reference_sha256=b.sha(R/'evidence/teammate_spectral.parquet'),reference_errors=errors,permutation_seed=20261127,paired_channels_permuted_together=True,daily_histograms_and_totals_preserved=True,source_function_128sample_reconstruction=True,known_weekly_peak_test=True,constant_series_missing_test=True,feature_hashes={k:b.sha(R/(k+'_features.parquet'))for k in frames},source_sha256=b.sha(Path(__file__)),seconds=time.time()-start),R/'BUILD_COMPLETE.json');print(json.dumps(dict(status='complete',eligible=int(enough.sum()),max_reference_error=max(errors.values()),seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
