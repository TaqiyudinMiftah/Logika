"""Owned VM bridge, exact packets, idempotent detached launch, bounded recovery."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tarfile,time,traceback
from transport import Colab,LOCAL
p=argparse.ArgumentParser();p.add_argument('--number',type=int,required=True);p.add_argument('--slots',type=int,default=1);a=p.parse_args()
L=LOCAL;N=a.number;P=L/f'private/vm{N}';P.mkdir(exist_ok=True);V=Colab(N);Q=f'GPU_VM{N}_QUEUE'
H='faris@100.68.29.126';R='/mnt/home/faris/logika-sixhour-broad-20261028';C=json.loads((L/'CLOCK.json').read_text());end=C['model_deadline_epoch']
def dump(v,n):
 (P/n).write_text(json.dumps(v,indent=2)+'\n')
def run(args,timeout=900):return subprocess.run(args,text=True,capture_output=True,check=True,timeout=timeout)
def lab(code,timeout=120):return subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=20',H,'python3 -'],input=code,text=True,capture_output=True,check=True,timeout=timeout)
def gates():
 d=P/'gates';d.mkdir(exist_ok=True)
 run(['rsync','-az','--include=GATE*.json','--exclude=*',H+':'+R+'/',str(d)+'/'],timeout=120)
 with tarfile.open(P/'gates.tar.gz','w:gz')as t:
  for f in d.glob('GATE*.json'):t.add(f,arcname=f.name)
 V.put(P/'gates.tar.gz',V.root+'/gates.tar.gz',timeout=120);V.ssh('tar -xzf '+V.root+'/gates.tar.gz -C '+V.root)
def collect():
 code="""from pathlib import Path
import tarfile,json,subprocess,time
r=Path('/root/logika-sixhour-broad-20261028')
with tarfile.open(r/'EVIDENCE.tar.gz','w:gz',compresslevel=1)as t:
 for n in ['runs','dispatch']:
  if(r/n).exists():t.add(r/n,arcname=n)
 for p in list(r.glob('*STATE.json'))+list(r.glob('*FINAL.json')):t.add(p,arcname=p.name)
 if(r/'ENVIRONMENT.json').exists():t.add(r/'ENVIRONMENT.json',arcname='ENVIRONMENT.json')
"""
 V.ssh('python3 -',code=code);V.get(V.root+'/EVIDENCE.tar.gz',P/'EVIDENCE.tar.gz',timeout=300)
 m=P/'mirror';m.mkdir(exist_ok=True)
 with tarfile.open(P/'EVIDENCE.tar.gz')as t:t.extractall(m,filter='data')
 # Every attempt is private until aggregate publisher sanitizes it.
 for n in ['runs','dispatch']:
  if(m/n).exists():run(['rsync','-az',str(m/n)+'/',H+':'+R+'/'+n+'/'],timeout=180)
 for f in list(m.glob('*STATE.json'))+list(m.glob('*FINAL.json')):
  run(['scp','-q',str(f),H+':'+R+'/'+f.name],timeout=90)
 # Bind contemporaneous physical CUDA measurements to worker START hostname.
 code="""from pathlib import Path
import json,hashlib
r=Path(ROOT_VALUE);p=r/'GPU_BACKENDS.json';v=json.loads(p.read_text())if p.exists()else{}
for f in(r/'dispatch').glob('*/attempt*/BACKEND.json'):
 b=json.loads(f.read_text())
 if b.get('GPU_name'):
  v[b['host']]=dict(host=b['host'],GPU_name=b['GPU_name'],GPU_memory_bytes=b['GPU_memory_bytes'],evidence_path=str(f.relative_to(r)),evidence_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),binding_basis='prelaunch same-host measured CUDA device')
t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\\n');t.replace(p)
""".replace('ROOT_VALUE',repr(R))
 lab(code)
 return json.loads((m/(Q+'_STATE.json')).read_text())if(m/(Q+'_STATE.json')).exists()else{}
def prepare():
 while time.time()<C['admission_end_epoch']:
  if V.json('SETUP_FAILURE.json'):raise RuntimeError('Owned runtime setup failure; private evidence retained')
  env=V.json('ENVIRONMENT.json')
  if env and env['status']=='PASS':break
  dump(dict(stage='environment_setup',epoch=time.time()),'CONTROLLER_STATE.json');time.sleep(20)
 else:raise TimeoutError('Original admission closed')
 dump(dict(stage='uploading_private_packets',epoch=time.time()),'CONTROLLER_STATE.json')
 V.put(L/'private/vm1/INPUTS.tar.gz',V.root+'/INPUTS.tar.gz',timeout=max(60,int(C['admission_end_epoch']-time.time()-60)))
 V.ssh('tar -xzf '+V.root+'/INPUTS.tar.gz -C '+V.root,timeout=300)
 with tarfile.open(P/'SOURCE.tar.gz','w:gz')as t:
  for directory in ['source','ops']:t.add(L/directory,arcname=directory,filter=lambda x:None if '__pycache__'in x.name else x)
  for f in list(L.glob('*.json'))+list(L.glob('*.jsonl')):t.add(f,arcname=f.name)
  t.add(L/'research/FEATURE_HYPOTHESES.json',arcname='research/FEATURE_HYPOTHESES.json')
  if N==6:t.add(L/'private/vm1/mirror/runs',arcname='runs')
 V.put(P/'SOURCE.tar.gz',V.root+'/SOURCE.tar.gz');V.ssh('tar -xzf '+V.root+'/SOURCE.tar.gz -C '+V.root)
 gates();assert V.json('GATE.json')['status']=='PASS'
 code="""from pathlib import Path
import json,os,subprocess
r=Path('/root/logika-sixhour-broad-20261028');name=QUEUE_VALUE;p=r/(name+'_DISPATCH.json');active=False
if p.exists():
 old=json.loads(p.read_text())
 try:os.kill(old['pid'],0);active=True
 except ProcessLookupError:pass
if active:print(json.dumps({'pid':old['pid'],'reused':True}))
else:
 with(r/(name+'.log')).open('ab')as log:
  command=[str(r/'env311/bin/python'),'-u',str(r/'ops/worker_queue.py'),'--resource','gpu','--slots',str(SLOTS_VALUE),'--name',name,'--owner',str(OWNER_VALUE)]
  process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,cwd=str(r))
 p.write_text(json.dumps({'pid':process.pid,'command':command})+'\\n');print(json.dumps({'pid':process.pid,'started':True}))
""".replace('QUEUE_VALUE',repr(Q)).replace('SLOTS_VALUE',str(a.slots)).replace('OWNER_VALUE',str(N))
 V.ssh('python3 -',code=code,timeout=60)
normal=False
try:
 prepare()
 failures=0
 while time.time()<end+30:
  try:
   s=collect();gates();dump(dict(stage='GPU_queue_running',epoch=time.time(),queue=s),'CONTROLLER_STATE.json');failures=0
   if V.json(Q+'_FINAL.json'):normal=True;break
  except Exception:
   failures+=1;(P/'COLLECTION_RETRY.txt').write_text(traceback.format_exc())
   dump(dict(stage='collection_retry',consecutive=failures,epoch=time.time()),'CONTROLLER_STATE.json')
   # Collection/ack errors never cancel healthy workers; retry under same clock.
  time.sleep(35)
 if time.time()>=end:normal=True
 collect()
except BaseException:
 (P/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());dump(dict(stage='needs_execution_reconciliation',epoch=time.time()),'CONTROLLER_STATE.json')
finally:
 released=False
 if normal or time.time()>=end:
  try:V.stop();released=True
  except Exception:(P/'RELEASE_FAILURE.txt').write_text(traceback.format_exc())
 if released:
  lab("from pathlib import Path\nimport json,time\np=Path("+repr(R+f'/GPU_COLLECTION_VM{N}_COMPLETE.json')+")\np.write_text(json.dumps({'epoch':time.time(),'released':True,'session':"+repr(V.session)+"})+'\\n')")
 dump(dict(stage='released'if released else'needs_reconciliation_watchdog_retains_deadline',epoch=time.time()),'FINAL_STATE.json')
