"""Host-local queues with canonical fitted-model retention and no score pruning."""
from pathlib import Path
import os,sys,time,json,subprocess,signal,fcntl,shutil
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'training-plan.json').read_text());D=p['original_deadline'];host=os.environ['CAMPAIGN_HOST'];jobs=p['host_jobs'][host]
lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert p['training_admitted']and json.loads((R/'RETENTION_GATE.json').read_text())['status']=='passed'
def write(x):
 t=R/'TRAINING_STATUS.tmp';t.write_text(json.dumps(x,indent=2));t.replace(R/'TRAINING_STATUS.json')
def terminate(a):
 os.killpg(a['process'].pid,signal.SIGTERM)
 try:return a['process'].wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(a['process'].pid,signal.SIGKILL);return a['process'].wait()
queue=[];completed=[];failed=[];active={};closed=False;durations=[]
for j in jobs:
 d=R/f'runs/job{j}'
 if(d/'complete.json').exists():completed.append(j)
 else:assert not d.exists(),'Partial job retained; explicit owner review required';queue.append(j)
while queue or active:
 now=time.time()
 for g,a in list(active.items()):
  code=a['process'].poll()
  if code is None and(now>=D or(R/'STOP').exists()or now-a['start']>3600):code=terminate(a)
  if code is not None:
   a['log'].close();del active[g]
   if code==0 and(R/f"runs/job{a['job']}/complete.json").exists():completed.append(a['job']);durations.append(now-a['start'])
   else:failed.append(dict(job=a['job'],exit_code=code));closed=True
 if now>=D-10800 or(R/'STOP').exists():closed=True
 for gpu in [0,1]:
  if closed or gpu in active or not queue:continue
  if host=='pc2' and gpu==0:
   reserved=subprocess.check_output(['systemctl','--user','show','logika-kumo-supervised-view-20261021.service','-p','MainPID'],text=True).strip().split('=')[-1]
   if reserved.isdigit()and int(reserved)>0 and Path('/proc/'+reserved).exists():continue
  estimate=1.5*max(durations)+120 if durations else 1800
  if now+estimate>=D-10800:continue
  if shutil.disk_usage(R).free<3*1024**3:continue
  available=int([l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')][0])*1024
  if available<10*1024**3:continue
  free,util=[int(v.strip())for v in subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
  if free<11000 or util>=10:continue
  j=queue.pop(0);log=(R/f'job{j}.log').open('ab',buffering=0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));child=subprocess.Popen([sys.executable,str(R/'source/worker.py'),'--job',str(j)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);active[g]=dict(job=j,process=child,start=time.time(),log=log)
 write(dict(status='running',host=host,completed=completed,failed=failed,queued=queue,active=[dict(gpu=g,job=a['job'],pid=a['process'].pid,started=a['start'])for g,a in active.items()],original_deadline=D,admissions_closed=closed))
 if closed and not active:break
 time.sleep(5)
write(dict(status='complete'if len(completed)==len(jobs)else'partial',host=host,completed=completed,failed=failed,unstarted=queue,original_deadline=D))
