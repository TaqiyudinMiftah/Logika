"""Verify immutable full inputs, then admit the already frozen recipe."""
from pathlib import Path
import json,hashlib,tarfile,time
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
p=json.loads((R/'plan.json').read_text());assert time.time()<p['original_deadline']-10800
for name in ['preflight.json','full_shape_preflight.json','pipeline_preflight.json']:assert json.loads((R/name).read_text())['status']=='passed'
a=R/'full-inputs.tar.gz';assert sha(a)=='10e1f8a356606a4e414321d58e6899771e111a9d4e8c028cc739462bb1d97520'
assert not(R/'full_data').exists()
with tarfile.open(a)as t:
 for m in t.getmembers():
  assert not Path(m.name).is_absolute()and '..'not in Path(m.name).parts
  assert m.isfile()or m.isdir()
  assert (R/m.name).resolve().is_relative_to(R.resolve())
 t.extractall(R)
seen=np.zeros(70000,int);checks=[]
for f in range(5):
 d=R/f'full_data/fold{f}';meta=json.loads((d/'manifest.json').read_text())
 for name,digest in meta['files'].items():assert sha(d/name)==digest
 x=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');ix=dict(np.load(d/'indices.npz'))
 assert len(x)==50000 and len(q)==20000 and'target'in x and'target'not in q
 assert len(np.load(d/'correction_labels.npy'))==6000 and len(ix['validation'])==14000
 assert len(set(x.application_id))==50000 and len(set(q.application_id))==20000 and set(x.application_id).isdisjoint(q.application_id)
 assert np.array_equal(np.sort(np.r_[ix['context'],ix['correction'],ix['validation']]),np.arange(70000))
 assert hashlib.sha256('\n'.join(x.application_id).encode()).hexdigest()==meta['context_ids_sha256']
 assert np.array_equal(np.load(d/'spline_validation.npz')['rows'],ix['validation']);seen[ix['validation']]+=1;checks.append(meta)
assert np.all(seen==1)
p['model_training_admitted']=True
p['phenotype_source_sha256']={s.name:sha(s)for s in sorted((R/'source').glob('*.py'))}
(R/'INPUT_ADMISSION.json').write_text(json.dumps(dict(status='passed',archive_sha256=sha(a),folds_verified=5,unique_validation_rows=70000,
 standard_CV_training_label_overlap_between_folds=True,current_fit_uses_only_current_context_and_correction_targets=True,
 historical_labels_reused=True,admitted_at=time.time()),indent=2))
(R/'plan.json').write_text(json.dumps(p,indent=2));print('ADMITTED',flush=True)
