"""Persistent host119 owns all eleven sessions, CPU queue, release and receipts."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,shlex,shutil,subprocess,sys,tarfile,time,traceback
R=Path('/home/faris/logika-twelvehour-next-20261029');CANON='/mnt/home/faris/logika-twelvehour-next-20261029';LAB='faris@100.68.29.126'
os.environ['PATH']='/home/faris/.local/bin:'+os.environ['PATH']
def dump(v,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def call(args,timeout=180):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
def push(p,dest=None):call(['scp','-q',str(p),LAB+':'+CANON+'/'+(dest or p.name)],120)
def unit(name,python,script,args,root,memory,quota,end):
 call(['systemd-run','--user','--quiet','--unit='+name,'--property=MemoryMax='+memory,'--property=CPUQuota='+quota,'--property=KillMode=control-group','--property=RuntimeMaxSec='+str(max(1,int(end-time.time()))),'--property=StandardOutput=append:'+str(root/'service.log'),'--property=StandardError=append:'+str(root/'service.log'),python,'-u',str(script),*args],60)
def cpu_collect(dest):
 upload=CANON+'/private/uploads/cpu119/runs';call(['ssh',LAB,'mkdir -p '+upload])
 call(['rsync','-az',str(dest/'runs')+'/',LAB+':'+upload+'/'],240)
 call(['ssh',LAB,'python3 '+CANON+'/ops/atomic_collect.py --owner cpu119 --source '+upload],240)
 for n in ['BACKEND_STATE.json','BACKEND_FINAL.json','OWNER_FAILURE.json']:
  if(dest/n).exists():push(dest/n,'CPU119_'+n)
 if(dest/'BACKEND_FINAL.json').exists():
  p=R/'BACKEND_COLLECTION_cpu119_COMPLETE.json';dump(dict(epoch=time.time(),owner='cpu119',collected=True,completed_fits=len(list((dest/'runs').glob('*/attempt1/RESULT.json')))),p);push(p)
def inventory(label):
 root=R/'private/resources';root.mkdir(exist_ok=True)
 def one(n):
  d=dict(account=n,epoch=time.time())
  for op in ['usage','sessions']:
   p=subprocess.run(['colab','--account',str(n),'--config',str(R/f'private/colab/account{n}/sessions.json'),op],capture_output=True,text=True,timeout=90);d[op]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
  dump(d,root/f'{label}_account{n}.json');return d
 with concurrent.futures.ThreadPoolExecutor(max_workers=11)as e:items=list(e.map(one,range(1,12)))
 dump(dict(epoch=time.time(),profiles=items),root/(label+'.json'));push(root/(label+'.json'),'RESOURCE_'+label+'.json')
def main():
 C=json.load(open(R/'CLOCK.json'));P=R/'private/colab';P.mkdir(parents=True,exist_ok=True)
 # Reuse byte-identical owned input packet archive without another large local copy.
 old=Path('/home/faris/logika-twelvehour-20261029/private/colab/PACKETS.tar')
 if not(P/'PACKETS.tar').exists():os.link(old,P/'PACKETS.tar')
 with tarfile.open(P/'SOURCE.tar.gz','w:gz')as tar:
  for name in ['source','ops','evidence','CONFIG.json','CLOCK.json','SOURCE_SEAL.json','VALIDATION_CONTRACT.json','INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','EXECUTION_PLAN.json','GATE.json','TRAINING_BRIEF.md','DEADLINE.md','TRAINING_STATE.json','AGENTS.md','PREPUBLICATION.json','RESOURCE_PREPUBLICATION.json','COLAB_RESOURCE_ADDENDUM.json','EXTRA_DESIGNS.jsonl']:
   tar.add(R/name,arcname=name,filter=lambda i:None if '__pycache__'in i.name else i)
 # Separate CPU root avoids queue-state collisions with controller metadata.
 dest=R/'private/backends/cpu119';dest.mkdir(parents=True,exist_ok=True)
 for p in R.glob('*'):
  if p.is_file()and p.suffix in ['.json','.jsonl','.md']:shutil.copy2(p,dest/p.name)
 for name in ['source','ops','evidence']:shutil.copytree(R/name,dest/name,dirs_exist_ok=True)
 for name in ['logs','runs','private']:(dest/name).mkdir(exist_ok=True)
 if not(dest/'packets').exists():
  packet_parent=R/'private/shared-packets';packet_parent.mkdir(exist_ok=True)
  with tarfile.open(old)as tar:tar.extractall(packet_parent,filter='data')
  (dest/'packets').symlink_to(packet_parent/'packets',target_is_directory=True)
 py='/home/faris/logika-twelvehour-20261029/env311/bin/python'
 unit('logika-next-cpu119-20261029',py,dest/'ops/owner_queue.py',['--owner','cpu119'],dest,'40G','1200%',C['model_deadline_epoch'])
 # Drive backup uses the explicit profile2 token on this personal control host only.
 unit('logika-next-drive-backup-20261029','/usr/bin/python3',R/'ops/drive_backup.py',[],R,'3G','100%',C['deadline_epoch'])
 call(['/usr/bin/python3','-u',str(R/'ops/allocate_colab_profiles.py')],600)
 for n in range(1,12):
  d=P/f'account{n}'
  if(d/'ALLOCATED_READY.json').exists():
   unit(f'logika-next-colab-a{n}-20261029','/usr/bin/python3',R/'ops/colab_backend_controller.py',['--account',str(n)],d,'2G','100%',C['deadline_epoch']-60)
  else:
   # Refusal/setup failure is a terminal resource outcome, never an implicit retry.
   value=dict(epoch=time.time(),account=n,collected=True,released=True,status='not_allocated_or_setup_failed',scientific_conclusion=False)
   if(d/'NEEDS_RECONCILIATION.json').exists():
    # Stop only the exact session named in this campaign's allocation intent.
    intent=json.load(open(d/'ALLOCATION_INTENT.json'));p=subprocess.run(['colab','--account',str(n),'--config',str(d/'sessions.json'),'stop','--session',intent['session']],capture_output=True,text=True,timeout=150);dump(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),d/'RECONCILIATION_STOP.json');value['released']=p.returncode==0
   dump(value,d/'FINAL_STATE.json')
   if value['released']:push(d/'FINAL_STATE.json',f'BACKEND_COLLECTION_account{n}_COMPLETE.json')
 last=0
 while time.time()<C['model_deadline_epoch']-120:
  try:cpu_collect(dest)
  except Exception:(R/'private/CPU_COLLECTION_FAILURE.log').write_text(traceback.format_exc())
  if time.time()-last>=1800:inventory('snapshot_'+str(int(time.time())));last=time.time()
  final=[n for n in range(1,12)if(P/f'account{n}/FINAL_STATE.json').exists()]
  dump(dict(epoch=time.time(),stage='remote_running',CPU_completed=len(list((dest/'runs').glob('*/attempt1/RESULT.json'))),terminal_accounts=final,all11profiles=True,laptop_required=False,automatic_refill=True,deadline_epoch=C['deadline_epoch']),R/'HOST_STATE.json');push(R/'HOST_STATE.json')
  if len(final)==11 and(dest/'BACKEND_FINAL.json').exists():break
  time.sleep(30)
 # Owned controllers have earlier collection cutoff and kernel-independent release.
 subprocess.run(['systemctl','--user','stop','logika-next-cpu119-20261029'],capture_output=True)
 cpu_collect(dest)
 for n in range(1,12):
  d=P/f'account{n}'
  if(d/'ALLOCATED_READY.json').exists()and not(d/'RELEASED.json').exists():
   intent=json.load(open(d/'ALLOCATION_INTENT.json'));p=subprocess.run(['colab','--account',str(n),'--config',str(d/'sessions.json'),'stop','--session',intent['session']],capture_output=True,text=True,timeout=150);dump(dict(returncode=p.returncode,owned_session=intent['session'],epoch=time.time()),d/'DEADLINE_STOP.json')
 inventory('FINAL_RESOURCE_AUDIT')
 dump(dict(epoch=time.time(),stage='terminal',all11profiles_checked=True,laptop_required=False,clock_sha256=hashlib.sha256((R/'CLOCK.json').read_bytes()).hexdigest()),R/'HOST_FINAL.json');push(R/'HOST_FINAL.json')
if __name__=='__main__':
 try:main()
 except BaseException:
  (R/'private/HOST_FAILURE.log').write_text(traceback.format_exc());dump(dict(epoch=time.time(),status='host_execution_review_required',independent_VM_deadline_watchdogs=True),R/'HOST_FAILURE.json')
  try:push(R/'HOST_FAILURE.json')
  except Exception:pass
  raise
