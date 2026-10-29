"""One preregistered diagnostic; unchanged training math, no query labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import hashlib,json,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import numpy as np,pandas as pd
import cpu_models
from common import load,dump,sha,utc,check_seal
clock=check_seal();d=R/'private/cpu-diagnostic';d.mkdir(exist_ok=False)
plan=load('CPU_DIAGNOSTIC_PLAN.json');assert plan['original_source_sha256']==sha(R/'source/cpu_models.py')
task=plan['selected_failed_task'];design=load('CONFIG.json')['designs'][task.split('__s')[0]]
p=R/'packets/fold0';m=json.load(open(p/'MANIFEST.json'));roles=np.load(p/'roles.npz')
frame=pd.read_parquet(p/'features.parquet',columns=m['views'][design['view']]);snapshots=[];original=cpu_models._ks;record={}
def capture(y,score):
 if sys._getframe(1).f_code.co_name=='metric':snapshots.append(np.asarray(score).copy())
 return original(y,score)
cpu_models._ks=capture
def trace(frame,event,arg):
 if event=='exception'and frame.f_code.co_name=='train_predict'and frame.f_code.co_filename==cpu_models.__file__ and isinstance(arg[1],AssertionError):
  v=frame.f_locals
  if 'stop_metric'in v and 'stop_probability'in v:
   best=v['selected_rounds'];native=snapshots[best-1];replay=v['stop_probability'];y=v['y'][v['stop']]
   np.savez_compressed(d/'stopping_replay_private.npz',native=native,replay=replay)
   record.update(status='diagnosed_original_guard_failure',selected_rounds=best,stored_native_stopping_KS=v['stop_metric'],recomputed_native_KS=original(y,native),python_margin_replay_KS=original(y,replay),max_abs_probability_error=float(np.max(np.abs(native-replay))),native_distinct_scores=len(np.unique(native)),replay_distinct_scores=len(np.unique(replay)),native_exact_stopping_KS_replay=abs(original(y,native)-v['stop_metric'])<1e-10,probability_error_below_original_5e_6=bool(np.max(np.abs(native-replay))<=5e-6))
 return trace
sys.settrace(trace);started=time.time()
try:
 spec={**design,'inner_split_seed':m['inner_split_seed'],'deadline_epoch':min(time.time()+180,clock['model_deadline_epoch']),'max_seconds':180}
 cpu_models.train_predict(frame,roles['context'],roles['y_context'],np.r_[roles['correction'],roles['validation']],spec,2026)
 record['unexpected_original_success']=True
except AssertionError as e:record['original_failure']=str(e)
finally:
 sys.settrace(None);record.update(UTC=utc(),seconds=time.time()-started,task=task,diagnostic_plan_sha256=sha(R/'CPU_DIAGNOSTIC_PLAN.json'),original_model_source_sha256=sha(R/'source/cpu_models.py'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),clock_sha256=sha(R/'CLOCK.json'),outer_labels_opened=False,model_selection_unchanged=True,weights_saved=False)
 dump(record,R/'CPU_REPLAY_DIAGNOSTIC.json');print(json.dumps(record))
