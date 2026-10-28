"""Independent retained-model content and raw-output audit; never scores partial folds."""
from pathlib import Path
import json,hashlib,joblib,numpy as np,pandas as pd
R=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
plan=json.loads((R/'training-plan.json').read_text());cols=json.loads((R/'columns.json').read_text());checks=[]
for d in sorted((R/'runs').glob('job*'),key=lambda p:int(p.name[3:])):
 if not(d/'complete.json').exists():continue
 j=int(d.name[3:]);done=json.loads((d/'complete.json').read_text());assert done['status']=='complete'and done['job']==j
 for n,h in done['files'].items():assert sha(d/n)==h
 source=R/f'data/job{j}';manifest=json.loads((source/'manifest.json').read_text())
 for n,h in manifest['files'].items():assert sha(source/n)==h
 c=pd.read_parquet(source/'context.parquet');q=pd.read_parquet(source/'query.parquet');model=joblib.load(d/'model.joblib');raw=np.load(d/'raw.npz');ix=np.load(source/'indices.npz');bound=json.loads((d/'boundary.json').read_text());role=manifest['role']
 assert model['kind']=='frozen_context_ICL_model_v1'and model['checkpoint_sha256']==plan['checkpoint_sha256']and model['columns']==cols and model['seed']==2026+role['outer_fold']
 assert model['engine_sha256']==sha(R/'source/gpu_backend.py')and model['model_code_sha256']==sha(R/'source/context_model.py')
 pd.testing.assert_frame_equal(model['context'],c[['application_id','target']+cols],check_exact=True)
 assert len(c)==role['context_rows']and len(q)==role['query_rows']and'target'not in q
 assert set(c.application_id).isdisjoint(q.application_id)and np.array_equal(q.application_id.to_numpy(str),raw['application_id'])and np.array_equal(ix['validation'],raw['rows'])
 assert not{'application_id','customer_id','target','split','row_id','row_index'}&set(cols)
 assert np.isfinite(raw['prediction']).all()and np.isfinite(raw['replayed_prediction']).all()
 replay=float(np.max(abs(raw['prediction']-raw['replayed_prediction'])));assert replay<2e-5 and abs(replay-bound['full_cold_model_replay_max_error'])<1e-12
 checks.append(dict(job=j,context_rows=len(c),query_rows=len(q),model_sha256=sha(d/'model.joblib'),full_saved_model_replay_max_error=replay,raw_outputs_sha256=sha(d/'raw.npz')))
out=dict(status='passed_for_collected_subset',collected_models_checked=len(checks),checks=checks,all30complete=len(checks)==30,partial_validation_scores_computed=False,scope='Independently re-read all retained model bytes, verify exact context data/labels/columns/seed/pinnedcode and every query-row raw cold-replay value. This doesnotdeclare a complete fitted recipe or new independentscore.')
(R/'COLLECTED_CONTENT_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in ['status','collected_models_checked','all30complete','partial_validation_scores_computed']}),flush=True)
