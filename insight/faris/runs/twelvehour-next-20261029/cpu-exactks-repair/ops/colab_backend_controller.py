"""Persistent-host controller for one already allocated, explicitly owned VM."""
from pathlib import Path
import argparse,hashlib,json,os,shlex,subprocess,sys,tarfile,time,traceback
R=Path('/home/faris/logika-twelvehour-next-20261029');VROOT='/root/logika-twelvehour-next-20261029';LAB='faris@100.68.29.126';CANON='/mnt/home/faris/logika-twelvehour-next-20261029'
os.environ['PATH']='/home/faris/.local/bin:'+os.environ['PATH']
a=argparse.ArgumentParser();a.add_argument('--account',type=int,required=True);N=a.parse_args().account;P=R/f'private/colab/account{N}';C=json.load(open(R/'CLOCK.json'));NAME=f'logika-next-a{N}-20261029';KEY='/home/faris/.ssh/id_ed25519_logika_lab'
PREFIX=['/home/faris/.local/bin/colab','--account',str(N),'--config',str(P/'sessions.json')]
assert json.load(open(P/'ALLOCATION_RESPONSE.json'))['returncode']==0 and not(P/'RELEASED.json').exists()
OPTS=['-o','BatchMode=yes','-o','ConnectTimeout=40','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(P/'known_hosts'),'-o','HostKeyAlias='+NAME,'-i',KEY,'-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+str(Path.home()/'.cache/logika-colab-control'/f'next-a{N}'),'-o','ProxyCommand='+shlex.join(PREFIX+['ssh','--proxy-mode','--session',NAME,'--identity',KEY])]
def run(args,timeout=180):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def ssh(command,code=None,timeout=120):return subprocess.run(['ssh','-T',*OPTS,'root@colab',command],input=code,check=True,capture_output=True,text=True,timeout=timeout)
def remote_json(name):
    code='from pathlib import Path;p=Path('+repr(VROOT+'/'+name)+');print(p.read_text()if p.exists()else "null")';return json.loads(ssh('python3 -',code).stdout)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(stage,**values):dump(P/'CONTROLLER_STATE.json',dict(epoch=time.time(),account=N,session=NAME,stage=stage,**values))
def collect():
    code='''from pathlib import Path
import tarfile
r=Path(ROOT)
with tarfile.open(r/'EVIDENCE.tar.gz','w:gz',compresslevel=1)as t:
 for base in ['runs','logs']:
  if(r/base).exists():
   for p in(r/base).rglob('*'):
    if p.is_file()and p.suffix in ['.json','.npz','.log']:t.add(p,arcname=str(p.relative_to(r)))
 for n in ['BACKEND_GATE.json','BACKEND_STATE.json','BACKEND_FINAL.json','ENVIRONMENT_LIVE.json','SETUP_FAILURE.json','QUEUE_DISPATCH.json']:
  if(r/n).exists():t.add(r/n,arcname=n)
'''.replace('ROOT',repr(VROOT))
    ssh('python3 -',code,timeout=180);run(['scp','-q',*OPTS,'root@colab:'+VROOT+'/EVIDENCE.tar.gz',str(P/'EVIDENCE.tar.gz')],timeout=300)
    mirror=P/'mirror';mirror.mkdir(exist_ok=True)
    with tarfile.open(P/'EVIDENCE.tar.gz')as t:t.extractall(mirror,filter='data')
    checked=0
    for p in(mirror/'runs').glob('*/attempt1/RESULT.json'):
        v=json.load(open(p));assert sha(p.parent/'prediction.npz')==v['prediction_sha256'];assert sha(p.parent/'START.json')==v['start_sha256'];assert v['weights_saved']is False;checked+=1
    (mirror/'runs').mkdir(exist_ok=True)
    if(mirror/'runs').exists():run(['ssh',LAB,'mkdir -p '+CANON+f'/private/uploads/account{N}/runs']);run(['rsync','-az','--ignore-existing',str(mirror/'runs')+'/',LAB+':'+CANON+f'/private/uploads/account{N}/runs/'],timeout=180)
    run(['ssh',LAB,'python3 '+CANON+'/ops/atomic_collect.py --owner account'+str(N)+' --source '+CANON+'/private/uploads/account'+str(N)+'/runs'],timeout=240)
    for n in ['BACKEND_GATE.json','BACKEND_STATE.json','BACKEND_FINAL.json','ENVIRONMENT_LIVE.json','SETUP_FAILURE.json']:
        p=mirror/n
        if p.exists():run(['rsync','-az',str(p),LAB+':'+CANON+f'/COLAB_A{N}_'+n],timeout=90)
    manifest={str(p.relative_to(mirror)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in mirror.rglob('*')if p.is_file()};dump(P/'PRIVATE_MANIFEST.json',manifest)
    dump(P/'COLLECTION_RECEIPT.json',dict(epoch=time.time(),checked_fit_records=checked,manifest_sha256=sha(P/'PRIVATE_MANIFEST.json'),bytes=sum(v['bytes']for v in manifest.values()),predictions_private=True));return checked

collected=False;released=False;normal=False
try:
    receipt=json.load(open(R/'RESOURCE_PREPUBLICATION.json'));assert len(receipt['remote_commit'])==40 and receipt['extra_designs_sha256']==sha(R/'EXTRA_DESIGNS.jsonl')
    state('uploading_sealed_source');ssh('mkdir -p '+VROOT)
    run(['scp','-q',*OPTS,str(R/'private/colab/SOURCE.tar.gz'),'root@colab:'+VROOT+'/SOURCE.tar.gz'],timeout=300)
    ssh('tar -xzf '+VROOT+'/SOURCE.tar.gz -C '+VROOT)
    code='''from pathlib import Path
import subprocess,sys,json
r=Path(ROOT);p=r/'BOOTSTRAP_DISPATCH.json'
assert not p.exists(),'No duplicate setup process'
with(r/'setup.log').open('ab')as f:
 q=subprocess.Popen([sys.executable,'-u',str(r/'ops/bootstrap_colab_owned.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
p.write_text(json.dumps({'pid':q.pid,'setup_only':True})+'\\n');print('bootstrap started')
'''.replace('ROOT',repr(VROOT))
    ssh('python3 -',code);state('uploading_hash_pinned_packets')
    run(['scp','-q',*OPTS,str(R/'private/colab/PACKETS.tar'),'root@colab:'+VROOT+'/PACKETS.tar'],timeout=min(1200,max(60,int(C['admission_end_epoch']-time.time()-60))))
    ssh('tar -xf '+VROOT+'/PACKETS.tar -C '+VROOT,timeout=300)
    while time.time()<C['admission_end_epoch']:
        if remote_json('SETUP_FAILURE.json'):raise RuntimeError('Pinned dependency/checkpoint setup failed; evidence retained')
        env=remote_json('ENVIRONMENT_LIVE.json')
        if env and env['status']=='PASS':dump(P/'ENVIRONMENT_LIVE.json',env);break
        state('environment_setup');time.sleep(20)
    else:raise TimeoutError('Original admission cutoff during setup')
    code='''from pathlib import Path
import hashlib,json,subprocess
r=Path(ROOT);p=r/'QUEUE_DISPATCH.json';assert not p.exists(),'Never duplicate scientific queue'
inputs=json.load(open(r/'INPUTS.json'))
for fold in range(5):
 d=r/f'packets/fold{fold}';m=json.load(open(d/'MANIFEST.json'));assert hashlib.sha256((d/'MANIFEST.json').read_bytes()).hexdigest()==inputs['packet_manifests'][str(fold)]
 for n,k in [('features.parquet','feature_sha256'),('roles.npz','roles_sha256'),('anchor.npz','anchor_sha256')]:assert hashlib.sha256((d/n).read_bytes()).hexdigest()==m[k]
(r/'VM_STARTED_TRAINING.json').write_text(json.dumps({'phase':'registered_fullsize_pilot','originalclockpreserved':True})+'\\n')
with(r/'backend_queue.log').open('ab')as f:q=subprocess.Popen([str(r/'env311/bin/python'),'-u',str(r/'ops/owner_queue.py'),'--owner',OWNER],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
p.write_text(json.dumps({'pid':q.pid,'owner':OWNER})+'\\n');print(json.dumps({'pid':q.pid}))
'''.replace('ROOT',repr(VROOT)).replace('OWNER',repr(str(N)))
    launch=json.loads(ssh('python3 -',code,timeout=180).stdout);dump(P/'CONTROLLER_TRAINING_STARTED.json',dict(epoch=time.time(),pid=launch['pid'],owner=N,original_clock_sha256=sha(R/'CLOCK.json')))
    state('registered_pilot_then_qualified_queue',pid=launch['pid'])
    failures=0
    while time.time()<C['model_deadline_epoch']-180:
        try:
            checked=collect();collected=True;failures=0
            final=remote_json('BACKEND_FINAL.json');gate=remote_json('BACKEND_GATE.json')
            state('running',checked_fit_records=checked,backend_gate=gate)
            if final or(gate and gate['status']!='PASS'):normal=True;break
        except Exception:
            failures+=1;(P/'COLLECTION_FAILURE_PRIVATE.log').write_text(traceback.format_exc());state('collection_retry',consecutive=failures)
        time.sleep(30)
    else:
        code='import os,signal,json;from pathlib import Path;p=Path('+repr(VROOT+'/QUEUE_DISPATCH.json')+');os.killpg(json.load(open(p))["pid"],signal.SIGTERM)';ssh('python3 -',code);normal=True
    collect();collected=True
except Exception:
    (P/'CONTROLLER_FAILURE_PRIVATE.log').write_text(traceback.format_exc());state('execution_review_required')
    try:collect();collected=True
    except Exception:(P/'FINAL_COLLECTION_FAILURE_PRIVATE.log').write_text(traceback.format_exc())
finally:
    p=subprocess.run([*PREFIX,'stop','--session',NAME],capture_output=True,text=True,timeout=150);dump(P/'RELEASE_RESPONSE.json',dict(epoch=time.time(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));released=p.returncode==0
    if released:dump(P/'RELEASED.json',dict(epoch=time.time(),session=NAME,owned_only=True))
    result=dict(epoch=time.time(),account=N,session=NAME,collected=collected,released=released,normal_queue_terminal=normal,manifest_sha256=sha(P/'PRIVATE_MANIFEST.json')if(P/'PRIVATE_MANIFEST.json').exists()else None,failed_attempts_preserved=True,original_deadline_epoch=C['deadline_epoch'])
    dump(P/'FINAL_STATE.json',result)
    if released and collected:
        run(['scp','-q',str(P/'FINAL_STATE.json'),LAB+':'+CANON+f'/BACKEND_COLLECTION_account{N}_COMPLETE.json'])
