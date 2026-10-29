"""One fixed ten-member refit, idle CUDA slots only, fresh bounded deadline."""
from pathlib import Path
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import read,dump,sha,utc
GPU='/mnt/home/faris/logika-model-arena-20261025/env311/bin/python';CPU='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python';cfg=read(R/'CONFIG.json');clock=read(R/'DEPLOYMENT_CLOCK.json');assert read(R/'PREFLIGHT_AUDIT.json')['status']=='PASS';assert sha(R/'CONFIG.json')==clock['config_sha256'];assert sha(R/'SOURCE_SEAL.json')==clock['source_seal_sha256'];assert len(read(R/'PREPUBLICATION.json')['remote_commit'])==40
active={};started=set();tasks=[(s,f)for s in cfg['seeds']for f in range(5)];completed=[];pilot=(2026,0);qualified=False
try:
 while time.time()<clock['deadline_epoch']-15:
  for gpu,(seed,fold,unit)in list(active.items()):
   state=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()
   if state not in ['active','activating','reloading']:
    d=R/f'predictions/s{seed}/fold{fold}';assert(d/'RESULT.json').exists(),f'Preserved failed member needs reconciliation:{seed}/{fold}'
    v=read(d/'RESULT.json');assert v['prediction_sha256']==sha(d/'prediction.npz')and v['audit']['first_batch_replay_error']<1e-7;del active[gpu]
  completed=[(s,f)for s,f in tasks if(R/f'predictions/s{s}/fold{f}/RESULT.json').exists()]
  if pilot in completed and not qualified:
   first=read(R/'predictions/s2026/fold0/RESULT.json');assert first['seconds']<900 and first['audit']['first_batch_replay_error']<1e-7;qualified=True;dump(dict(status='PASS',UTC=utc(),pilot_seconds=first['seconds'],first_batch_replay_error=first['audit']['first_batch_replay_error'],model_recipe_unchanged=True),R/'DEPLOYMENT_RUNTIME_GATE.json')
  if len(completed)==10:break
  if not qualified and (pilot in started or active):time.sleep(3);continue
  available=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.splitlines()
  for line in available:
   gpu,memory,util=[int(x.strip())for x in line.split(',')]
   if gpu in active or memory>512 or util>5:continue
   pending=[t for t in tasks if t not in completed and t not in started]
   if not pending:continue
   seed,fold=pending[0];assert qualified or(seed,fold)==pilot;assert not(R/f'predictions/s{seed}/fold{fold}').exists();unit=f'logika-submit25-s{seed}-f{fold}-20261029.service'
   command=['systemd-run','--user','--quiet','--unit='+unit,'--property=MemoryMax=20G','--property=CPUQuota=200%','--property=RuntimeMaxSec='+str(min(1200,max(1,int(clock['deadline_epoch']-time.time()-10)))),'--setenv=CUDA_VISIBLE_DEVICES='+str(gpu),'--property=StandardOutput=append:'+str(R/'logs'/f's{seed}_f{fold}.log'),'--property=StandardError=append:'+str(R/'logs'/f's{seed}_f{fold}.log'),GPU,'-u',str(R/'source/predict.py'),'--fold',str(fold),'--seed',str(seed)];subprocess.run(command,check=True,capture_output=True,text=True);active[gpu]=(seed,fold,unit);started.add((seed,fold))
   if not qualified:break
  dump(dict(stage='refitting_fixed_winner',UTC=utc(),completed=len(completed),expected=10,active={str(k):v for k,v in active.items()},deadline_epoch=clock['deadline_epoch'],no_paid_allocation=True),R/'LIVE_STATE.json');time.sleep(3)
 assert len(completed)==10,'Bounded fixed deployment ended incomplete; no submission'
 subprocess.run([CPU,str(R/'source/assemble.py')],cwd=R,check=True,timeout=180);dump(dict(stage='audited_CSV_ready',UTC=utc(),completed=10,submission_audit_sha256=sha(R/'SUBMISSION_AUDIT.json')),R/'LIVE_STATE.json')
except BaseException:
 dump(dict(stage='needs_execution_review',UTC=utc(),private_diagnostic=traceback.format_exc()),R/'QUEUE_FAILURE.json');raise
finally:
 for gpu,(seed,fold,unit)in active.items():subprocess.run(['systemctl','--user','stop',unit],capture_output=True)
