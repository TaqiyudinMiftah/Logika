"""Extend frozen feature formulas to test; prove old membership parity first."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import gc,json
import numpy as np
import pandas as pd
from common import R,P,A,PROD,read,sha,dump,utc,sources
from feature_formulas import Features,ratio

class ProductionBank(Features):
 def __init__(self,fold,context,parity=False):
  old=Path(read(P/'PATHS.json')['horizontal']);bank=Path(read(P/'PATHS.json')['bank'])
  self.manifest=read(P/'BANK_MANIFEST.json');self.lookup={n:i for i,n in enumerate(self.manifest['columns'])}
  self.numeric=np.load(bank/'numeric.npy',mmap_mode='r');self.cats=pd.read_parquet(bank/'categorical.parquet')
  receipt=read(PROD/'prior70k/COMPLETE.json');prior_path=PROD/'prior70k/fold0/features.parquet'
  assert sha(prior_path)==receipt['features_sha256']
  full=pd.read_parquet(prior_path);head_path=PROD/'features70/expanded.parquet'
  assert sha(head_path)==read(PROD/'features70/COMPLETE.json')['head_frame_sha256']
  head=pd.read_parquet(head_path)
  assert full.application_id.tolist()==head.application_id.tolist() and len(full)==100000
  self.ids=full.application_id.to_numpy(str)
  for c in head:
   if c not in ['application_id','customer_id']:full[c]=head[c].to_numpy()
  prior=pd.read_parquet(old/f'packets/fold{fold}/current.parquet')
  assert set(prior).issubset(full.columns)
  current=full[list(prior)].copy()
  if parity:current=pd.concat([prior,current.iloc[70000:]],ignore_index=True)
  for c in current:current[c]=current[c].astype(prior[c].dtype)
  self.current=current
  previous=pd.read_parquet(old/'extra.parquet');extra={}
  for path,columns in read(old/'EXTRA_SOURCES.json').items():
   table=pd.read_parquet(path).set_index('application_id').loc[self.ids]
   for c in columns:
    if c in previous:extra[c]=table[c].to_numpy()
  assert set(previous).issubset(extra)
  self.extra=pd.DataFrame(extra)[list(previous)]
  if parity:self.extra=pd.concat([previous,self.extra.iloc[70000:]],ignore_index=True)
  for c in self.extra:self.extra[c]=self.extra[c].astype(previous[c].dtype)
  self.ctx=np.asarray(context,dtype=np.int64);self.cache={};self.prior_cache={}
  self.legacy={p['name']:p for p in self.manifest['priors']if p.get('legacy')}
  self.priors={}
  for p in self.manifest['priors']:
   self.priors[p['name']]=(p,False)
   if p.get('sd'):self.priors[p['sd']]=(p,True)
  # Match the sealed campaign's explicit removal of cached cohort priors.
  # Their sufficient statistics must use the requested training membership.
  self.current=self.current.drop(columns=sorted(set(self.current)&set(self.priors)))
  assert len(self.cats)==len(self.numeric)==len(self.current)==len(self.extra)==100000
 def base(self,name):
  if name in self.cache:return self.cache[name]
  if name in self.current:value=self.current[name].to_numpy()
  elif name in self.priors:
   p,uncertainty=self.priors[name];k,n=self.base(p['k']),self.base(p['n']);mu=self.cohort(p['k'],p['n']);strength=p.get('strength',24)
   value=ratio(np.nan_to_num(k)+strength*mu,np.nan_to_num(n)+strength)
   if uncertainty:value=np.sqrt(np.maximum(value*(1-value)/(np.nan_to_num(n)+strength+1),0))
  elif name in self.extra:value=self.extra[name].to_numpy()
  elif name in self.lookup:value=np.asarray(self.numeric[:,self.lookup[name]],dtype=float)
  elif name in self.cats:value=self.cats[name].to_numpy()
  else:raise KeyError(name)
  self.cache[name]=value;return value
 def compute(self,s):
  if s['op']=='monthly_spend_ratio':return (ratio(self.base(s['inputs'][0]),self.base(s['inputs'][1]))*(30/s['window_days'])).astype(np.float32)
  return super().compute(s)

def frame(bank,columns,specs):
 result=pd.DataFrame({name:bank.compute(specs[name])for name in columns})
 for c in result:
  if pd.api.types.is_numeric_dtype(result[c]):result[c]=result[c].replace([np.inf,-np.inf],np.nan).astype(np.float32)
 assert 'target'not in result and result.shape==(100000,99)
 return result

def main():
 config=read(R/'CONFIG.json');assert config['refit_context_rows']==70000
 assert read(P/'ARTIFACT_CONSISTENCY_AUDIT.json')['status']=='PASS'
 assert read(P/'FINAL_SELECTION.json')['identity']['cohort']==config['cohort']
 specs={s['id']:s for s in read(P/'FEATURE_SCHEMA.json')['specs']};y=pd.read_parquet(A/'ranking-pipeline-20261020/input/data/train_features.parquet',columns=['application_id','target'])
 reports=[]
 for fold in range(5):
  out=R/f'packets/fold{fold}';assert not out.exists();out.mkdir(parents=True)
  manifest=read(P/f'packets/fold{fold}/MANIFEST.json');columns=manifest['views']['global99'];roles=np.load(P/f'packets/fold{fold}/roles.npz')
  inner=np.load(P/f'packets/fold{fold}/inner_roles.npz');print(json.dumps({'stage':'feature_parity','fold':fold}),flush=True)
  # The declared inner-role arrays are global row indices.
  print(json.dumps({'inner_role_fields':inner.files}),flush=True)
  from sklearn.model_selection import train_test_split
  ii,_=train_test_split(np.arange(50000),test_size=.15,stratify=roles['y_context'],random_state=20261029+fold)
  bank=ProductionBank(fold,roles['context'][ii],parity=True);candidate=frame(bank,columns,specs)
  reference=pd.read_parquet(P/f'packets/fold{fold}/features.parquet',columns=columns)
  mismatches=[]
  for c in columns:
   a=candidate[c].iloc[:70000].to_numpy();b=reference[c].to_numpy()
   equal=np.array_equal(a,b,equal_nan=True)if np.issubdtype(a.dtype,np.number)else pd.Series(a).fillna('__NA__').equals(pd.Series(b).fillna('__NA__'))
   if not equal:mismatches.append(c)
  assert not mismatches,('Frozen feature parity failed',fold,mismatches)
  del bank,candidate,reference;gc.collect()
  bank=ProductionBank(fold,np.arange(70000));production=frame(bank,columns,specs)
  assert np.array_equal(bank.ids[:70000],y.application_id.to_numpy(str))
  sample=pd.read_csv(A/'exploration-12h-20261017/raw/sample_submission(3).csv')
  assert np.array_equal(bank.ids[70000:],sample.application_id.to_numpy(str))
  production.to_parquet(out/'features.parquet',index=False);np.save(out/'y_context.npy',y.target.to_numpy(np.int8));np.save(out/'test_ids.npy',bank.ids[70000:])
  report=dict(status='PASS',fold=fold,rows=100000,context_rows=70000,test_rows=30000,columns=columns,feature_sha256=sha(out/'features.parquet'),labels_sha256=sha(out/'y_context.npy'),test_ids_sha256=sha(out/'test_ids.npy'),original_manifest_sha256=sha(P/f'packets/fold{fold}/MANIFEST.json'),original_training_feature_parity='bit-exact float32 on all70000 original rows under original inner membership',preprocessing_refit='Same formulas and fixed columns; cohort statistics use all70k permitted training covariates.',source_sha256=sources(),no_query_labels=True)
  dump(report,out/'MANIFEST.json');reports.append(report)
  print(json.dumps({'stage':'production_packet_ready','fold':fold}),flush=True);del bank,production;gc.collect()
 dump(dict(status='PASS',UTC=utc(),folds=reports),R/'PREPARATION_AUDIT.json')
if __name__=='__main__':main()
