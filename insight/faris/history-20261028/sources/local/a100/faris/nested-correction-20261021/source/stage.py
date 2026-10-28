"""Build supervised GPU packets from verified inner/final covariate worlds."""
from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];A=R.parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
p=json.loads((R/'plan.json').read_text());assert json.loads((R/'INDEPENDENT_AUDIT.json').read_text())['status']=='passed'
base=pd.read_parquet(Path(p['input'])/'data/train_features.parquet',columns=['application_id','target']);lock=json.loads((Path(p['input'])/'BASELINE_LOCK.json').read_text());cols=lock['Kumo']['features'];extra=['risk_conditional_negative_months','risk_conditional_nonnegative_months','aux_account_months','aux_orders90','aux_ecommerce_observed_days','aux_allocation_count'];head=lock['correction']['features']+extra
assert len(cols)==90 and len(head)==18
counts_path=A/'coverage-transfer-20261021/data/features.parquet';counts=pd.read_parquet(counts_path).set_index('application_id');assert counts.index.is_unique
results=[]
for j in range(30):
 job=p['jobs'][str(j)];src=R/f'fold{j}';c=json.loads((src/'complete.json').read_text());assert sha(src/'features.parquet')==c['feature_table_sha256']
 frame=pd.read_parquet(src/'features.parquet').set_index('application_id');f=frame.loc[base.application_id].copy();assert np.array_equal(f.index,base.application_id)
 for n in extra:f[n]=counts[n].reindex(f.index).to_numpy()
 ix=dict(np.load(Path(p['input'])/f'artifacts/fold{j}/indices.npz'));ci,vi=ix['context'],ix['validation'];assert len(ci)==job['context_rows']and len(vi)==job['query_rows']and set(ci).isdisjoint(vi)
 context=f.iloc[ci][cols].copy();context.insert(0,'application_id',base.application_id.iloc[ci].to_numpy());context['target']=base.target.iloc[ci].to_numpy(np.int64)
 query=f.iloc[vi][list(dict.fromkeys(cols+head))].copy();query.insert(0,'application_id',base.application_id.iloc[vi].to_numpy());assert'target'not in query and set(context.application_id).isdisjoint(query.application_id)
 dest=R/f'data/job{j}';assert not dest.exists();dest.mkdir(parents=True)
 context.to_parquet(dest/'context.parquet',index=False);query.to_parquet(dest/'query.parquet',index=False);np.savez_compressed(dest/'indices.npz',**ix)
 meta=dict(status='complete',job=j,role=job,feature_prior_context_ids_sha256=c['fit_ids_sha256'],context_ids_sha256=hashlib.sha256('\n'.join(context.application_id).encode()).hexdigest(),query_ids_sha256=hashlib.sha256('\n'.join(query.application_id).encode()).hexdigest(),source_features_sha256=c['feature_table_sha256'],rowlocal_coverage_sha256=sha(counts_path),query_labels_staged=False,files={z.name:sha(z)for z in dest.iterdir()if z.is_file()});assert meta['context_ids_sha256']==meta['feature_prior_context_ids_sha256'];(dest/'manifest.json').write_text(json.dumps(meta,indent=2));results.append(dict(job=j,manifest_sha256=sha(dest/'manifest.json')))
(R/'columns.json').write_text(json.dumps(cols,indent=2));(R/'head_columns.json').write_text(json.dumps(head,indent=2));(R/'DATA_READY.json').write_text(json.dumps(dict(status='complete',jobs=30,files=results,outer_labels_staged=False),indent=2))
