"""Personal-host owned Colab production controller, checkpoint and release."""
from pathlib import Path
import argparse,hashlib,json,os,shlex,subprocess,sys,tarfile,time,traceback
R=Path('/home/faris/logika-overnight-refit-20261030');VR='/root/logika-overnight-refit-20261030';os.environ['PATH']='/home/faris/.local/bin:'+os.environ['PATH'];LAB='faris@100.68.29.126';CAN='/mnt/home/faris/logika-overnight-refit-20261030'
p=argparse.ArgumentParser();p.add_argument('--account',type=int,required=True);N=p.parse_args().account;P=R/f'private/colab/account{N}';NAME=f'logika-0630-a{N}-20261030';KEY='/home/faris/.ssh/id_ed25519_logika_lab';PREFIX=['/home/faris/.local/bin/colab','--account',str(N),'--config',str(P/'sessions.json')];C=json.load(open(R/'CLOCK.json'))
OPTS=['-o','BatchMode=yes','-o','ConnectTimeout=30','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=2','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(P/'known_hosts'),'-o','HostKeyAlias='+NAME,'-i',KEY,'-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+str(Path.home()/'.cache/logika-colab-control'/f'0630-a{N}'),'-o','ProxyCommand='+shlex.join(PREFIX+['ssh','--proxy-mode','--session',NAME,'--identity',KEY])]
def run(args,**kw):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=kw.pop('timeout',180),**kw)
def ssh(cmd,code=None,timeout=180):return run(['ssh','-T',*OPTS,'root@colab',cmd],input=code,timeout=timeout)
def dump(n,v):(P/n).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def collect():
 code="from pathlib import Path;import tarfile;r=Path("+repr(VR)+");t=tarfile.open(r/'evidence.tgz','w:gz');[(t.add(p,arcname=str(p.relative_to(r))))for base in ['predictions','logs']for p in(r/base).rglob('*')if p.is_file()and p.suffix in ['.json','.npz','.log']];t.close()"
 ssh('python3 -',code);run(['scp','-q',*OPTS,'root@colab:'+VR+'/evidence.tgz',str(P/'evidence.tgz')],timeout=300);mirror=P/'mirror';mirror.mkdir(exist_ok=True)
 with tarfile.open(P/'evidence.tgz')as t:t.extractall(mirror,filter='data')
 checked=[]
 for p in(mirror/'predictions').glob('*/s*/fold*/RESULT.json'):
  v=json.load(open(p));assert sha(p.parent/'prediction.npz')==v['prediction_sha256']and sha(p.parent/'START.json')==v['start_sha256'];checked.append(str(p.relative_to(mirror)))
 # Publish only complete directories, atomically on the canonical host.
 for relative in checked:
  d=mirror/relative;target=CAN+'/'+str(d.parent.relative_to(mirror));run(['ssh',LAB,'mkdir -p '+target+'.upload'])
  run(['rsync','-az',str(d.parent)+'/',LAB+':'+target+'.upload/'],timeout=180)
  verify="from pathlib import Path;import hashlib,json,os;p=Path("+repr(target+'.upload')+");v=json.load(open(p/'RESULT.json'));assert hashlib.sha256((p/'prediction.npz').read_bytes()).hexdigest()==v['prediction_sha256'];assert hashlib.sha256((p/'START.json').read_bytes()).hexdigest()==v['start_sha256'];q=Path("+repr(target)+");p.rename(q)if not q.exists()else None"
  run(['ssh',LAB,'python3 -'],input=verify)
 dump('COLLECTION.json',dict(epoch=time.time(),complete=len(checked),private_prediction_hashes_verified=True));return len(checked)
assert(P/'ALLOCATED_READY.json').exists()and not(P/'RELEASED.json').exists()
try:
 ssh('mkdir -p '+VR);run(['scp','-q',*OPTS,str(R/'private/colab/REFIT_BUNDLE.tgz'),'root@colab:'+VR+'/bundle.tgz'],timeout=600);ssh('tar -xzf '+VR+'/bundle.tgz -C '+VR,timeout=180)
 ssh('python3 -',"import subprocess,sys;from pathlib import Path;r=Path("+repr(VR)+");f=(r/'setup.log').open('ab');p=subprocess.Popen([sys.executable,'-u',str(r/'ops/bootstrap_colab_owned.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)")
 while time.time()<C['admission_end_epoch']:
  q=ssh('python3 -',"from pathlib import Path;import json;r=Path("+repr(VR)+");print(json.dumps({'pass':(r/'ENVIRONMENT_LIVE.json').exists(),'fail':(r/'SETUP_FAILURE.json').exists()}))");v=json.loads(q.stdout)
  if v['fail']:raise RuntimeError('Pinned setup failed')
  if v['pass']:break
  time.sleep(15)
 else:raise TimeoutError('Setup admission cutoff')
 tasks=[('kumo70',s,f)for s in [2026,7731]for f in range(5)]if N==2 else[('kumo25',s,f)for i,(s,f)in enumerate((s,f)for s in[2026,7731]for f in range(5))if i%4==[8,9,10,11].index(N)]
 code="""import json,subprocess,sys,time
from pathlib import Path
r=Path(ROOT);(r/'VM_STARTED_TRAINING.json').write_text(json.dumps({'epoch':time.time()})+'\\n')
for kind,seed,fold in TASKS:
 with(r/(kind+'_'+str(seed)+'_'+str(fold)+'.log')).open('ab')as f:
  q=subprocess.run([str(r/'env311/bin/python'),'-u',str(r/'source/refit.py'),'--kind',kind,'--seed',str(seed),'--fold',str(fold)],stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 if q.returncode:break
(r/'REFIT_QUEUE_FINAL.json').write_text(json.dumps({'epoch':time.time(),'tasks':TASKS})+'\\n')
""".replace('ROOT',repr(VR)).replace('TASKS',repr(tasks))
 ssh('python3 -',"from pathlib import Path;Path("+repr(VR+'/ops/vm_queue.py')+").write_text("+repr(code)+")")
 ssh('python3 -',"import subprocess;from pathlib import Path;r=Path("+repr(VR)+");f=(r/'queue.log').open('ab');p=subprocess.Popen([str(r/'env311/bin/python'),'-u',str(r/'ops/vm_queue.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)")
 while time.time()<C['model_deadline_epoch']-60:
  n=collect();dump('CONTROLLER_STATE.json',dict(epoch=time.time(),complete=n,tasks=tasks,stage='fresh_refit'))
  final=ssh('python3 -',"from pathlib import Path;print(int(Path("+repr(VR+'/REFIT_QUEUE_FINAL.json')+").exists()))").stdout.strip()
  if final=='1':break
  time.sleep(30)
 collect()
except Exception:
 (P/'CONTROLLER_FAILURE_PRIVATE.log').write_text(traceback.format_exc())
 try:collect()
 except Exception:(P/'COLLECTION_FAILURE_PRIVATE.log').write_text(traceback.format_exc())
finally:
 q=subprocess.run([*PREFIX,'stop','--session',NAME],capture_output=True,text=True,timeout=120);dump('RELEASE_RESPONSE.json',dict(epoch=time.time(),returncode=q.returncode,stdout=q.stdout,stderr=q.stderr))
 if q.returncode==0:dump('RELEASED.json',dict(epoch=time.time(),owned_only=True))
