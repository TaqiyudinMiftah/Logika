"""Recover the existing full fold-local view; do not fit across evaluation rows."""
from pathlib import Path
import json,time,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];A=R.parent;OLD=A/'fold-local-priors-20261021';STAGED=A/'kumo-temperature-adaptation-20261021/full_data'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
assert json.loads((OLD/'INDEPENDENT_AUDIT.json').read_text())['status']=='passed'
assert time.time()<1792585871.823748-10800
EXTRA=['risk_conditional_negative_months','risk_conditional_nonnegative_months','aux_account_months','aux_orders90','aux_ecommerce_observed_days','aux_allocation_count']
schemas=[]
for f in range(5):
 dest=R/f'data/fold{f}';assert not dest.exists();dest.mkdir(parents=True)
 src=OLD/f'fold{f}';done=json.loads((src/'complete.json').read_text());assert sha(src/'features.parquet')==done['feature_table_sha256']
 broad=src/'workspace/data/processed/application_features_v5.parquet';full=pd.read_parquet(broad)
 assert full.target.eq(0).all(),'Feature builders had only placeholder targets'
 for n in ['application_id','customer_id']:assert full[n].is_unique
 full=full.set_index('application_id');small=pd.read_parquet(src/'features.parquet').set_index('application_id').reindex(full.index)
 for n in small.columns:
  if n in full:
   a=full[n].replace([np.inf,-np.inf],np.nan);b=small[n]
   pd.testing.assert_series_equal(a,b,check_dtype=False,check_names=False,check_categorical=False,check_exact=True)
  else:full[n]=small[n]
 s=STAGED/f'fold{f}';context=pd.read_parquet(s/'context.parquet',columns=['application_id','target']+EXTRA);q=pd.read_parquet(s/'query.parquet',columns=['application_id']+EXTRA);ix=dict(np.load(s/'indices.npz'));ci,hi,vi=[ix[k]for k in ['context','correction','validation']]
 assert len(ci)==50000 and len(hi)==6000 and len(vi)==14000 and len(set(np.r_[ci,hi,vi]))==70000
 assert set(full.index[full.split.eq('train')])==set(context.application_id)
 assert hashlib.sha256('\n'.join(context.application_id).encode()).hexdigest()==done['fit_ids_sha256']
 apps=np.full(70000,'',object);apps[ci]=context.application_id;apps[np.r_[hi,vi]]=q.application_id
 assert len(set(apps))==70000 and not any(v==''for v in apps)
 drop={'customer_id','target','split','decision_datetime','application_id','row_id','row_index','index'}
 cols=[n for n in full if n not in drop and not pd.api.types.is_datetime64_any_dtype(full[n])]
 assert not any(any(k in n.lower()for k in ['rng','seed','hidden_label','row_index'])for n in cols)
 cats=full[cols].select_dtypes(include=['object','string','category']).columns.tolist()
 x=full.loc[apps,cols].reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
 for n in EXTRA:
  values=np.full(70000,np.nan);values[ci]=context[n].to_numpy();values[np.r_[hi,vi]]=q[n].to_numpy();x[n]=values
 cols=list(dict.fromkeys(cols+EXTRA))
 compact=list(dict.fromkeys([n for n in small.columns if n not in drop]+EXTRA));assert set(compact)<=set(cols)
 for n in cats:x[n]=x[n].astype('string').fillna('MISSING').astype(str)
 nums=[n for n in cols if n not in cats];x[nums]=x[nums].astype(np.float32)
 tr=np.r_[ci,hi];train=x.iloc[tr].reset_index(drop=True);train['target']=np.r_[context.target.to_numpy(np.int8),np.load(s/'correction_labels.npy')]
 train['application_id']=apps[tr];valid=x.iloc[vi].reset_index(drop=True);valid['application_id']=apps[vi]
 assert set(train.application_id).isdisjoint(valid.application_id)and'target'not in valid
 train.to_parquet(dest/'train.parquet',index=False);valid.to_parquet(dest/'validation.parquet',index=False);np.savez_compressed(dest/'indices.npz',context=ci,correction=hi,train=tr,validation=vi)
 schema=dict(columns=cols,categorical=cats,compact_columns=compact,numeric_dtype='float32');schemas.append(schema)
 write(dest/'schema.json',schema)
 write(dest/'manifest.json',dict(fold=f,context_rows=50000,inner_selection_rows=6000,final_training_rows=56000,outer_rows=14000,
  feature_prior_fit_IDs_sha256=done['fit_ids_sha256'],source_broad_sha256=sha(broad),source_compact_sha256=done['feature_table_sha256'],
  vendor_source_sha256=done['source_sha256'],labels_available_to_feature_estimation=False,constant_placeholder_target_removed=True,
  raw_snapshot_point_in_time_certified=False,full_view_feature_count=len(cols),files={p.name:sha(p)for p in dest.iterdir()if p.is_file()}))
 print(json.dumps(dict(fold=f,features=len(cols),cats=len(cats),status='staged')),flush=True)
assert all(s==schemas[0]for s in schemas)
write(R/'STAGING.json',dict(status='complete',folds=5,features=len(schemas[0]['columns']),compact_features=len(schemas[0]['compact_columns']),categorical=schemas[0]['categorical'],validation_targets_exported=False))
