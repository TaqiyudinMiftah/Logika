"""Workstation bridge: one quota-granted VM, exact packets, incremental evidence."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys,tarfile,time,traceback
from transport import Colab,LOCAL
L=LOCAL;P=L/'private/vm1';P.mkdir(exist_ok=True);V=Colab(1)
HOST='faris@100.68.29.126';REMOTE='/mnt/home/faris/logika-sixhour-broad-20261028';C=json.loads((L/'CLOCK.json').read_text());end=C['model_deadline_epoch']
def dump(v,n):
 (P/n).write_text(json.dumps(v,indent=2)+'\n')
def run(args,timeout=900):return subprocess.run(args,text=True,capture_output=True,check=True,timeout=timeout)
def lab(code,timeout=120):return run(['ssh','-oBatchMode=yes','-oConnectTimeout=20',HOST,'python3 -'],timeout=timeout)if code is None else subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=20',HOST,'python3 -'],input=code,text=True,capture_output=True,check=True,timeout=timeout)
def collect():
 code="""from pathlib import Path
import tarfile
r=Path('/root/logika-sixhour-broad-20261028')
with tarfile.open(r/'EVIDENCE.tar.gz','w:gz',compresslevel=1)as t:
 for name in ['runs','dispatch']:
  d=r/name
  if d.exists():t.add(d,arcname=name)
 for p in r.glob('*STATE.json'):t.add(p,arcname=p.name)
 for name in ['ENVIRONMENT.json','GPU_QUEUE_FINAL.json','SETUP_FAILURE.json']:
  p=r/name
  if p.exists():t.add(p,arcname=name)
"""
 V.ssh('python3 -',code=code);V.get(V.root+'/EVIDENCE.tar.gz',P/'EVIDENCE.tar.gz',timeout=300)
 mirror=P/'mirror';mirror.mkdir(exist_ok=True)
 with tarfile.open(P/'EVIDENCE.tar.gz')as t:t.extractall(mirror,filter='data')
 run(['rsync','-az',str(mirror)+'/',HOST+':'+REMOTE+'/'],timeout=300)
 return json.loads((mirror/'GPU_QUEUE_STATE.json').read_text())if(mirror/'GPU_QUEUE_STATE.json').exists()else{}
def gates():
 folder=P/'gates';folder.mkdir(exist_ok=True)
 run(['rsync','-az','--include=GATE*.json','--exclude=*',HOST+':'+REMOTE+'/',str(folder)+'/'],timeout=120)
 with tarfile.open(P/'gates.tar.gz','w:gz')as t:
  for p in folder.glob('GATE*.json'):t.add(p,arcname=p.name)
 V.put(P/'gates.tar.gz',V.root+'/gates.tar.gz',timeout=120);V.ssh('tar -xzf '+V.root+'/gates.tar.gz -C '+V.root)
try:
 # Bootstrap was explicitly dispatched by the human-authorized root. No new VM.
 while time.time()<C['admission_end_epoch']:
  value=V.json('ENVIRONMENT.json')
  if value and value.get('status')=='PASS':break
  if V.json('SETUP_FAILURE.json'):raise RuntimeError('Owned VM setup failed; inspect retained private setup evidence')
  dump(dict(stage='environment_setup',epoch=time.time()),'CONTROLLER_STATE.json');time.sleep(20)
 else:raise TimeoutError('Environment setup exceeded original admission clock')
 # This download is private and excludes the >2k full bank and evaluator labels.
 code="""from pathlib import Path
import json,tarfile
r=Path('REMOTE_VALUE')
if not(r/'PREPARATION_AUDIT.json').exists():print('pending')
else:
 out=r/'COLAB_COMPACT_INPUTS.tar.gz'
 if not out.exists():
  with tarfile.open(out,'w:gz',compresslevel=1)as t:
   for f in range(5):
    for name in ['features.parquet','roles.npz','inner_roles.npz','MANIFEST.json']:
     p=r/f'packets/fold{f}'/name;t.add(p,arcname=str(p.relative_to(r)))
 print(json.dumps({'bytes':out.stat().st_size,'ready':True}))
""".replace('REMOTE_VALUE',REMOTE)
 while time.time()<C['admission_end_epoch']:
  result=lab(code,timeout=900).stdout.strip()
  if result!='pending':dump(json.loads(result),'INPUT_BUNDLE.json');break
  dump(dict(stage='waiting_preparation',epoch=time.time()),'CONTROLLER_STATE.json');time.sleep(20)
 else:raise TimeoutError('Packets not ready within admission clock')
 dump(dict(stage='downloading_compact_packets',epoch=time.time()),'CONTROLLER_STATE.json')
 run(['scp','-q',HOST+':'+REMOTE+'/COLAB_COMPACT_INPUTS.tar.gz',str(P/'INPUTS.tar.gz')],timeout=max(60,int(C['admission_end_epoch']-time.time()-60)))
 dump(dict(stage='uploading_compact_packets',bytes=(P/'INPUTS.tar.gz').stat().st_size,epoch=time.time()),'CONTROLLER_STATE.json')
 V.put(P/'INPUTS.tar.gz',V.root+'/INPUTS.tar.gz',timeout=max(60,int(C['admission_end_epoch']-time.time()-60)))
 V.ssh('tar -xzf '+V.root+'/INPUTS.tar.gz -C '+V.root,timeout=180)
 bundle=P/'SOURCE.tar.gz'
 with tarfile.open(bundle,'w:gz')as t:
  for directory in ['source','ops']:t.add(L/directory,arcname=directory,filter=lambda x:None if '__pycache__'in x.name else x)
  for path in L.glob('*.json'):
   if path.name not in ['ENVIRONMENT.json']:t.add(path,arcname=path.name)
  t.add(L/'research/FEATURE_HYPOTHESES.json',arcname='research/FEATURE_HYPOTHESES.json')
 V.put(bundle,V.root+'/SOURCE.tar.gz');V.ssh('tar -xzf '+V.root+'/SOURCE.tar.gz -C '+V.root)
 while time.time()<C['admission_end_epoch']:
  gates();data=V.json('GATE.json')
  if data and data['status']=='PASS':break
  if data and data['status']=='FAIL':raise RuntimeError('Data gate failed; no GPU fits admitted')
  time.sleep(15)
 command='cd '+V.root+' && nohup env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '+V.root+'/env311/bin/python -u ops/worker_queue.py --resource gpu --slots 2 --name GPU_QUEUE > GPU_QUEUE.log 2>&1 < /dev/null &'
 V.ssh(command,timeout=60)
 dump(dict(stage='GPU_queue_running',epoch=time.time()),'CONTROLLER_STATE.json')
 while time.time()<end+30:
  state=collect();gates();dump(dict(stage='GPU_queue_running',epoch=time.time(),queue=state),'CONTROLLER_STATE.json')
  if V.json('GPU_QUEUE_FINAL.json'):break
  time.sleep(35)
 collect()
except BaseException:
 (P/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());dump(dict(stage='execution_failure',epoch=time.time(),traceback_sha256=hashlib.sha256((P/'CONTROLLER_FAILURE.txt').read_bytes()).hexdigest()),'CONTROLLER_STATE.json')
 try:collect()
 except BaseException:pass
finally:
 # The workstation and notebook-kernel watchdog both release only this owned VM.
 try:V.stop();released=True
 except BaseException:
  released=False;(P/'RELEASE_FAILURE.txt').write_text(traceback.format_exc())
 try:lab("from pathlib import Path\nimport json,time\np=Path("+repr(REMOTE+'/GPU_COLLECTION_COMPLETE.json')+")\np.write_text(json.dumps({'epoch':time.time(),'released':"+repr(released)+",'session':'logika-sixhour-20261028-1'})+'\\n')")
 except BaseException:pass
 dump(dict(stage='released'if released else'release_needs_attention',epoch=time.time()),'FINAL_STATE.json')
