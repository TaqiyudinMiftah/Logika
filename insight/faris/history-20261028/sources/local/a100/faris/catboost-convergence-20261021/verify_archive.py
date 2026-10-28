from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
from catboost import CatBoostClassifier
R=Path(__file__).resolve().parent
checks=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for view in ['full','compact']:
 for f in range(5):
  d=R/f'runs/{view}/folds/fold{f}';c=json.loads((d/'complete.json').read_text());assert c['status']=='complete'
  for n,h in c['files'].items():assert sha(d/n)==h
  schema=json.loads((R/f'data/fold{f}/schema.json').read_text());cols=schema['columns'if view=='full'else'compact_columns'];v=pd.read_parquet(R/f'data/fold{f}/validation.parquet');z=np.load(d/'validation.npz');assert np.array_equal(v.application_id.to_numpy(str),z['application_id'])
  errors=[]
  for file,key in [('model.cbm','prediction'),('matched_300_prefix.cbm','matched_300_prediction')]:
   m=CatBoostClassifier();m.load_model(str(d/file));p=m.predict_proba(v[cols],thread_count=2)[:,1];err=float(np.max(abs(p-z[key])));assert err<1e-14;errors.append(err)
  checks.append(dict(view=view,fold=f,full_hashes_verified=len(c['files']),full_and_prefix_replay_max_error=errors,outer_rows=len(v)))
(R/'ARCHIVE_AUDIT.json').write_text(json.dumps(dict(status='passed',checks=checks,trained_pipeline_count=2,outer_fold_jobs=10,actual_inner_plus_refit_count=20,derived_prefixes=10),indent=2));print('ALL_TWENTY_CHECKPOINT_REPLAYS_PASSED')
