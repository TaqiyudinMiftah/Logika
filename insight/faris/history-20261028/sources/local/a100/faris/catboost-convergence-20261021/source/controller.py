"""Two GPU slots, two fixed views, no partial-fold performance selection."""
from pathlib import Path
import os,fcntl,json,time,subprocess,sys,signal,shutil
R=Path(__file__).resolve().parents[1];p=json.loads((R/'plan.json').read_text());D=p['original_deadline'];V=['full','compact']
lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert p['training_admitted']
import hashlib
for name,expected in p['source_sha256'].items():assert hashlib.sha256((R/'source'/name).read_bytes()).hexdigest()==expected
for v in V:assert json.loads((R/f'PREFLIGHT_{v}.json').read_text())['status']=='passed'
def write(path,x):
 t=Path(path).with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(path)
def end(a):
 os.killpg(a['process'].pid,signal.SIGTERM)
 try:return a['process'].wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(a['process'].pid,signal.SIGKILL);return a['process'].wait()
queue=[];outcomes={};active={};durations={v:[]for v in V};stop_admission=False
for f in range(5):
 for v in V:
  d=R/f'runs/{v}/folds/fold{f}';key=f'{v}/{f}'
  if(d/'complete.json').exists():outcomes[key]='retained_complete'
  else:assert not d.exists(),'Partial attempt preserved; explicit review needed';queue.append((v,f))
while queue or active:
 now=time.time()
 # Sampling is observational and limited to campaign PIDs.
 try:
  lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines();mem={int(l.split(',')[0]):int(l.split(',')[1])for l in lines if','in l and l.split(',')[1].strip().isdigit()}
 except Exception:mem={}
 for g,a in list(active.items()):
  code=a['process'].poll();pid=a['process'].pid
  try:
   st=Path(f'/proc/{pid}/status').read_text().splitlines();rss=max([int(l.split()[1])for l in st if l.startswith('VmHWM:')]+[0]);a['peak_RAM_KiB']=max(a['peak_RAM_KiB'],rss)
  except FileNotFoundError:pass
  a['observed_peak_VRAM_MiB']=max(a['observed_peak_VRAM_MiB'],mem.get(pid,0))
  if code is None and(now>=D or(R/'STOP').exists()or now-a['started']>5400):code=end(a)
  if code is not None:
   ok=code==0 and(R/f"runs/{a['view']}/folds/fold{a['fold']}/complete.json").exists();key=f"{a['view']}/{a['fold']}";outcomes[key]='complete'if ok else'failed_or_partial'
   (R/'resources').mkdir(exist_ok=True);write(R/f"resources/{a['view']}_fold{a['fold']}.json",dict(pid=pid,gpu=g,seconds=now-a['started'],exit_code=code,peak_RAM_KiB=a['peak_RAM_KiB'],observed_peak_VRAM_MiB=a['observed_peak_VRAM_MiB']))
   if ok:durations[a['view']].append(now-a['started'])
   else:stop_admission=True
   a['log'].close();del active[g]
 if(R/'STOP').exists()or now>=D-10800:stop_admission=True
 for g in [0,1]:
  if stop_admission or g in active or not queue:continue
  if shutil.disk_usage(R).free<2*1024**3:continue
  if (R/'runs').exists() and sum(x.stat().st_size for x in (R/'runs').rglob('*') if x.is_file())>8*1024**3:continue
  free,util=[int(x.strip())for x in subprocess.check_output(['nvidia-smi',f'--id={g}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
  if free<11000 or util>=10:continue
  admitted=None
  for i,(v,f)in enumerate(queue):
   estimate=1.5*max(durations[v])+120 if durations[v]else(5400 if v=='full' else 1800)
   if now+estimate<D-10800:admitted=i;break
  if admitted is None:continue
  v,f=queue.pop(admitted);log=(R/f'{v}_fold{f}.log').open('ab',buffering=0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g))
  child=subprocess.Popen([sys.executable,str(R/'source/worker.py'),'fold','--fold',str(f),'--view',v],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  active[g]=dict(process=child,view=v,fold=f,started=time.time(),log=log,peak_RAM_KiB=0,observed_peak_VRAM_MiB=0)
 state=dict(status='running',original_deadline=D,queued=[dict(view=v,fold=f)for v,f in queue],outcomes=outcomes,active=[dict(gpu=g,pid=a['process'].pid,view=a['view'],fold=a['fold'],started=a['started'])for g,a in active.items()],admissions_closed=stop_admission)
 write(R/'STATUS.json',state)
 if stop_admission and not active:break
 time.sleep(5)
complete=[v for v in V if all((R/f'runs/{v}/folds/fold{f}/complete.json').exists()for f in range(5))]
if complete:
 with(R/'evaluation.log').open('ab',buffering=0)as log:code=subprocess.call([sys.executable,str(R/'source/report.py')],stdout=log,stderr=subprocess.STDOUT)
else:code=None
write(R/'STATUS.json',dict(status='complete'if len(complete)==2 and code==0 else'partial',completed_views=complete,outcomes=outcomes,unstarted=[dict(view=v,fold=f)for v,f in queue],report_exit_code=code,original_deadline=D))
