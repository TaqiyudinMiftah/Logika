"""Two owned idle-GPU slots, fixed ten-member recipe, then audited assembly."""
from pathlib import Path
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import read,dump,sha,utc
PY='/mnt/home/faris/logika-model-arena-20261025/env311/bin/python'
CPU='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
config=read(R/'CONFIG.json');tasks=[(s,f)for s in config['seeds']for f in range(5)]
first=read(R/'predictions/s2026/fold0/RESULT.json')
assert first['status']=='complete' and first['audit']['first_batch_replay_error']<1e-7
active={};started=set();limit=time.time()+1800
try:
 while time.time()<limit:
  for gpu,(seed,fold,unit)in list(active.items()):
   p=R/f'predictions/s{seed}/fold{fold}'
   state=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()
   if state in ['inactive','failed']:
    assert (p/'RESULT.json').exists(),f'Preserved failed worker requires reconciliation: {unit}'
    result=read(p/'RESULT.json');assert result['prediction_sha256']==sha(p/'prediction.npz')
    del active[gpu]
  completed=[(s,f)for s,f in tasks if (R/f'predictions/s{s}/fold{f}/RESULT.json').exists()]
  if len(completed)==len(tasks):break
  available=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.splitlines()
  for line in available:
   gpu,memory,util=[int(x.strip())for x in line.split(',')]
   if gpu in active or memory>512 or util>5:continue
   pending=[(s,f)for s,f in tasks if(s,f)not in completed and(s,f)not in started]
   if not pending:continue
   seed,fold=pending[0];path=R/f'predictions/s{seed}/fold{fold}';assert not path.exists()
   unit=f'logika-submit-best-s{seed}-f{fold}-20261029.service'
   command=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=20G','--property=CPUQuota=200%','--property=RuntimeMaxSec=900','--setenv=CUDA_VISIBLE_DEVICES='+str(gpu),'--working-directory='+str(R),PY,'-u','source/predict.py','--fold',str(fold),'--seed',str(seed)]
   subprocess.run(command,check=True,capture_output=True,text=True)
   active[gpu]=(seed,fold,unit);started.add((seed,fold))
  dump(dict(stage='predicting',UTC=utc(),completed=len(completed),expected=len(tasks),active={str(k):v for k,v in active.items()},no_paid_allocation=True),R/'LIVE_STATE.json')
  time.sleep(3)
 assert len(completed)==10,'Bounded submission queue ended; preserve partial results'
 subprocess.run([CPU,'source/assemble.py'],cwd=R,check=True,timeout=180)
 dump(dict(stage='prediction_complete_ready_to_submit',UTC=utc(),completed=10,submission_audit_sha256=sha(R/'SUBMISSION_AUDIT.json')),R/'LIVE_STATE.json')
except BaseException:
 dump(dict(stage='needs_execution_review',UTC=utc(),private_diagnostic=traceback.format_exc()),R/'QUEUE_FAILURE.json');raise
