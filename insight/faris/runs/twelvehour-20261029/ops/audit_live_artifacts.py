"""Nonterminal raw-artifact preflight. No model selection, labels or training."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import numpy as np
from common import load,sha,ids_sha,dump,utc,check_seal,tasks,catalog
check_seal();designs=catalog();completed=[];missing=[];roles_cache={}
try:
 for task,expected in tasks().items():
  d=R/f'runs/{task}/attempt1'
  if not(d/'RESULT.json').exists():missing.append(task);continue
  start=load(d/'START.json');result=load(d/'RESULT.json');fold=expected['fold']
  if fold not in roles_cache:
   with np.load(R/f'packets/fold{fold}/roles.npz')as a:roles_cache[fold]={k:a[k]for k in ['context','correction','validation']}
  roles=roles_cache[fold]
  assert result['status']=='complete'and start['task']==expected and start['seed']==expected['seed']+fold
  assert start['source_sha256']==sha(R/'source/worker.py')and sha(d/'START.json')==result['start_sha256']and sha(d/'prediction.npz')==result['prediction_sha256']
  with np.load(d/'prediction.npz')as raw:
   assert np.array_equal(raw['rows'],np.r_[roles['correction'],roles['validation']])and raw['raw'].shape==(20000,)and np.isfinite(raw['raw']).all()and np.all((raw['raw']>=0)&(raw['raw']<=1))
  assert start['context_rows_sha256']==ids_sha(roles['context'])and start['outer_labels_loaded']is False and start['correction_labels_used_for_training']is False
  assert result['weights_saved']is False and result['fitted_weights_saved']is False and result['KV_cache_saved']is False
  if designs[expected['design']]['resource']=='cpu':
   a=result['audit'];assert a['inner_train_rows_sha256']==start['declared_inner_train_rows_sha256']and a['stopping_rows_sha256']==start['declared_inner_stop_rows_sha256']and a['preprocessing_refit_on_context']is False
  else:
   a=result['audit'];assert not a['context_selection_uses_query_labels']and a['checkpoint_sha256']and all(x['engine_audit']['first_batch_replay_error']<1e-7 for x in a['bags'])
  completed.append(dict(task=task,result_sha256=sha(d/'RESULT.json'),prediction_sha256=sha(d/'prediction.npz')))
 dump(dict(status='PASS',UTC=utc(),completed_fits_checked=len(completed),not_completed_at_snapshot=len(missing),checks='Exact source/result/prediction hashes, permitted context/query rows, base+fold seeds, preprocessing and training boundaries, model batch replay and no weights/KV dumps',completed=completed,clock_sha256=sha(R/'CLOCK.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),nonterminal=True,outer_target_values_read=False,independent_generalization_evidence=False),R/'LIVE_RAW_ARTIFACT_PREFLIGHT.json')
 print('PASS',len(completed),'complete fits; no outer target values read')
except Exception:
 (R/'private/LIVE_PREFLIGHT_FAILURE_PRIVATE.log').write_text(traceback.format_exc());dump(dict(status='FAIL_REQUIRES_REVIEW',UTC=utc(),passed_before_failure=len(completed),trace_sha256=sha(R/'private/LIVE_PREFLIGHT_FAILURE_PRIVATE.log'),science_unchanged=True),R/'LIVE_RAW_ARTIFACT_PREFLIGHT.json');raise
