"""Prospective CPU repair, strictly inside the immutable parent deadline."""
import os
for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_key]='2'
from pathlib import Path
import fcntl,json,os,shlex,shutil,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];P=R.parent;sys.path.insert(0,str(R/'source'));sys.path.insert(0,str(R/'ops'))
from common import load,dump,sha,utc,check_seal,tasks
from atomic_collect import publish
PY='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
TORCH1='/mnt/home/faris/logika-model-arena-20261025/env311/bin/python'
TORCH3='/mnt/home/faris/logika-submit-best-sub25k-20261029/env_gpu/bin/python'
KEY='/mnt/home/faris/.ssh/id_ed25519_logika_backup119'
SPECS={'cpu2001':dict(ssh=[],root=str(R/'private/backends/cpu2001'),python=TORCH1,packets=str(P/'packets'),memory='28G'),
 'cpu2003':dict(ssh=['ssh','-o','BatchMode=yes','-p2003','faris@gpu-labkc.ub.ac.id'],root=str(R/'private/backends/cpu2003'),python=TORCH3,packets='/mnt/home/faris/logika-twelvehour-20261029/packets',memory='28G'),
 'cpu119':dict(ssh=['ssh','-o','BatchMode=yes','-i',KEY,'faris@100.89.211.119'],root='/home/faris/logika-next-obltree-20261029/owner119',python='/home/faris/logika-next-neural-cann-20261029/env311/bin/python',packets='/home/faris/logika-twelvehour-next-20261029/private/shared-packets/packets',memory='40G'),
 'cpu42':dict(ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-i',KEY,'faris@100.98.169.42'],root='/home/faris/logika-next-obltree-20261029/owner42',python='/home/faris/logika-next-neural-cann-20261029/env311/bin/python',packets='/home/faris/logika-next-cpu-exactks-20261029/shared-packets/packets',memory='40G')}
FILES=['CONFIG.json','SOURCE_SEAL.json','VALIDATION_CONTRACT.json','INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','EXECUTION_PLAN.json','CLOCK.json','GATE.json','TRAINING_BRIEF.md','DEADLINE.md','TRAINING_STATE.json','AGENTS.md','PREPUBLICATION.json','EXTRA_DESIGNS.jsonl']
def call(args,timeout=180):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
def module(name,extra=()):
 with(R/'logs'/f'{name}.log').open('ab')as f:subprocess.run([PY,str(R/'source'/f'{name}.py'),*extra],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,load('CLOCK.json')['deadline_epoch']-time.time()-10))
def setup(owner):
 s=SPECS[owner];dest=Path(s['root']);ssh=s['ssh'];slots=load('EXECUTION_PLAN.json')['owners'][owner]['slots']
 if ssh:
  call([*ssh,'mkdir','-p',str(dest)])
  transport=shlex.join(ssh[:-1]);target=ssh[-1]+':'+str(dest)+'/'
  for n in [*FILES,'source','ops','evidence']:call(['rsync','-az','-e',transport,str(R/n),target],240)
  code="from pathlib import Path;r=Path("+repr(str(dest))+");p=Path("+repr(s['packets'])+");assert p.is_dir();(r/'packets').symlink_to(p,target_is_directory=True)if not(r/'packets').exists()else None;[(r/n).mkdir(exist_ok=True)for n in ['private','logs','runs']]"
  call([*ssh,'python3 -c '+shlex.quote(code)])
 else:
  dest.mkdir(parents=True,exist_ok=True)
  for n in FILES:shutil.copy2(R/n,dest/n)
  for n in ['source','ops','evidence']:shutil.copytree(R/n,dest/n,dirs_exist_ok=True)
  if not(dest/'packets').exists():(dest/'packets').symlink_to(s['packets'],target_is_directory=True)
  for n in ['private','logs','runs']:(dest/n).mkdir(exist_ok=True)
 cmd=['systemd-run','--user','--quiet','--unit=logika-next-obltree-'+owner+'-20261029','--property=MemoryMax='+s['memory'],'--property=CPUQuota='+str(slots*200)+'%','--property=KillMode=control-group','--property=RuntimeMaxSec='+str(int(load('CLOCK.json')['model_deadline_epoch']-time.time()-20)),'--property=StandardOutput=append:'+str(dest/'service.log'),'--property=StandardError=append:'+str(dest/'service.log'),s['python'],'-u',str(dest/'ops/owner_queue.py'),'--owner',owner]
 call([*ssh,*cmd],60)
 dump(dict(UTC=utc(),owner=owner,root=str(dest),started=True,derived_clock_sha256=sha(R/'CLOCK.json')),R/f'DISPATCH_{owner}.json')
def collect(owner):
 s=SPECS[owner];ssh=s['ssh'];src=s['root']+'/runs'
 if ssh:src=ssh[-1]+':'+src;os.environ['LOGIKA_COLLECT_RSH']=shlex.join(ssh[:-1])
 try:receipt=publish(src,owner)
 finally:os.environ.pop('LOGIKA_COLLECT_RSH',None)
 metadata=R/'private/owner_metadata'/owner;metadata.mkdir(parents=True,exist_ok=True)
 if ssh:call(['rsync','-az','-e',shlex.join(ssh[:-1]),'--include=/BACKEND_*.json','--include=/OWNER_*.json','--exclude=*',ssh[-1]+':'+s['root']+'/',str(metadata)+'/'],90)
 else:
  for p in Path(s['root']).glob('BACKEND_*.json'):shutil.copy2(p,metadata/p.name)
 if(metadata/'BACKEND_FINAL.json').exists():dump(dict(UTC=utc(),owner=owner,receipt=receipt,terminal_sha256=sha(metadata/'BACKEND_FINAL.json')),R/f'COLLECTED_{owner}.json')
 return receipt
def main():
 for n in ['private','logs','runs','evaluation_inputs']:(R/n).mkdir(exist_ok=True)
 lock=(R/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 parent=load(P/'CLOCK.json');assert sha(P/'CLOCK.json')==load('PARENT_CLOCK_REFERENCE.json')['sha256']
 receipt=load('PREPUBLICATION.json');assert receipt['config_sha256']==sha(R/'CONFIG.json')and receipt['source_seal_sha256']==sha(R/'SOURCE_SEAL.json')
 existing_clock=(R/'CLOCK.json').exists()
 assert not existing_clock or '--resume' in sys.argv,'Never reset delegated deadline without explicit execution resume'
 for n in ['TRAINING_BRIEF.md','DEADLINE.md','TRAINING_STATE.json']:shutil.copy2(P/n,R/n)
 clock=dict(parent,config_sha256=sha(R/'CONFIG.json'),contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),execution_plan_sha256=sha(R/'EXECUTION_PLAN.json'),parent_clock_sha256=sha(P/'CLOCK.json'),derived_budget_only=True)
 if existing_clock:
  assert load('CLOCK.json')==clock,'Delegated clock cannot change during resume'
 else:dump(clock,R/'CLOCK.json');os.chmod(R/'CLOCK.json',0o444)
 if not(R/'packets').exists():(R/'packets').symlink_to(P/'packets',target_is_directory=True)
 for n in ['baseline.npz','previous.npz','target.npy']:shutil.copy2(P/'evaluation_inputs'/n,R/'evaluation_inputs'/n)
 if not(R/'GATE.json').exists():module('gate')
 assert load('GATE.json')['status']=='PASS'
 # Setups are independent and do not modify model/selection state.
 for owner in SPECS:
  if not(R/f'DISPATCH_{owner}.json').exists():setup(owner)
 last=0
 while time.time()<clock['model_deadline_epoch']-90:
  errors={}
  for owner in SPECS:
   try:collect(owner)
   except Exception as e:errors[owner]=type(e).__name__
  if time.time()-last>60:module('compose');module('evaluate');last=time.time()
  dump(dict(stage='running',UTC=utc(),completed_fits=len(list((R/'runs').glob('*/attempt1/RESULT.json'))),failed_attempts=len(list((R/'runs').glob('*/attempt1/FAILURE.json'))),planned_fits=len(tasks()),errors=errors,parent_clock_sha256=sha(P/'CLOCK.json'),original_deadline_epoch=clock['deadline_epoch'],source_version='Sparse_oblivious_tree_CANN'),R/'LIVE_STATE.json')
  if all((R/f'COLLECTED_{o}.json').exists()for o in SPECS):break
  time.sleep(15)
 for owner,s in SPECS.items():
  subprocess.run([*s['ssh'],'systemctl','--user','stop','logika-next-obltree-'+owner+'-20261029'],capture_output=True)
  try:collect(owner)
  except Exception:pass
 module('compose');module('evaluate',['--final']);module('audit')
 result=dict(stage='complete',UTC=utc(),completed_fits=len(list((R/'runs').glob('*/attempt1/RESULT.json'))),failed_attempts=len(list((R/'runs').glob('*/attempt1/FAILURE.json'))),planned_fits=len(tasks()),parent_clock_sha256=sha(P/'CLOCK.json'),derived_clock_sha256=sha(R/'CLOCK.json'),audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json'),separate_source_version_not_merged=True)
 dump(result,R/'COMPLETION.json');dump(result,R/'LIVE_STATE.json')
if __name__=='__main__':
 try:main()
 except BaseException:
  p=R/'private/COORDINATOR_FAILURE.log';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc());dump(dict(status='execution_review_required',UTC=utc(),trace_sha256=sha(p)),R/'COORDINATOR_FAILURE.json');raise
