"""Two independent GPU slots, immutable folds and inherited absolute deadline."""
import os,fcntl,subprocess,time,json,signal,sys,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
D=1792585871.823748;reserve=10800;active={};outcomes={}
def write(x):
 p=R/'STATUS.tmp';p.write_text(json.dumps(x,indent=2));p.replace(R/'STATUS.json')
assert json.loads((R/'plan.json').read_text())['training_admitted']is True
assert json.loads((R/'PREFLIGHT_REPR.json').read_text())['status']=='passed'
queue=[]
for f in range(5):
 d=R/f'folds/fold{f}'
 if (d/'complete.json').exists():outcomes[f]='retained_complete'
 else:assert not d.exists(),'Partial fold preserved; explicit review required';queue.append(f)
while queue or active:
 now=time.time()
 for gpu,a in list(active.items()):
  code=a['process'].poll()
  if code is None and (now>=D or (R/'STOP').exists() or now-a['started']>5400):
   os.killpg(a['process'].pid,signal.SIGTERM)
   try:code=a['process'].wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(a['process'].pid,signal.SIGKILL);code=a['process'].wait()
  if code is not None:
   outcomes[a['fold']]='complete'if code==0 and(R/f"folds/fold{a['fold']}/complete.json").exists()else'failed_or_partial'
   a['log'].close();del active[gpu]
   if code!=0:(R/'ADMISSION_FAILURE').write_text('Infrastructure failure; stop further admissions and preserve active jobs')
 if queue and(now>=D-reserve or(R/'STOP').exists()or(R/'ADMISSION_FAILURE').exists()):break
 for gpu in [0,1]:
  if gpu in active or not queue:continue
  if shutil.disk_usage(R).free<20*1024**3:continue
  owned=sum(x.stat().st_size for x in(R/'folds').rglob('*')if x.is_file())if(R/'folds').exists()else 0
  if owned>30*1024**3:continue
  raw=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
  if int(raw[0])<11000 or int(raw[1])>=10:continue
  f=queue.pop(0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));log=(R/f'fold{f}.log').open('ab',buffering=0)
  child=subprocess.Popen([sys.executable,str(R/'source/experiment.py'),'fold','--fold',str(f)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  active[gpu]=dict(process=child,fold=f,started=time.time(),log=log)
 write(dict(status='running',queued=queue,outcomes=outcomes,original_deadline=D,active=[dict(gpu=g,pid=a['process'].pid,fold=a['fold'],started=a['started'])for g,a in active.items()]))
 time.sleep(5)
# If admissions close, let already admitted jobs finish under their original cap.
for gpu,a in active.items():
 remaining=max(1,min(D-time.time(),5400-(time.time()-a['started'])))
 try:code=a['process'].wait(timeout=remaining)
 except subprocess.TimeoutExpired:
  os.killpg(a['process'].pid,signal.SIGTERM)
  try:code=a['process'].wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(a['process'].pid,signal.SIGKILL);code=a['process'].wait()
 outcomes[a['fold']]='complete'if code==0 and(R/f"folds/fold{a['fold']}/complete.json").exists()else'failed_or_partial';a['log'].close()
complete=all((R/f'folds/fold{f}/complete.json').exists()for f in range(5))
write(dict(status='training_complete'if complete else'partial',outcomes=outcomes,unstarted=queue,original_deadline=D))
if complete:
 with (R/'evaluation.log').open('ab',buffering=0)as log:code=subprocess.call([sys.executable,str(R/'source/report.py')],stdout=log,stderr=subprocess.STDOUT)
 write(dict(status='complete'if code==0 else'evaluation_failed',outcomes=outcomes,original_deadline=D))
