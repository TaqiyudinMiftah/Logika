"""One explicitly owned Colab session; resume control without repeating fits."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import traceback

R = Path('/home/faris/logika-midnight-multiseed-20261030')
VROOT = '/root/logika-midnight-multiseed-20261030'
LAB = 'faris@100.68.29.126'
CANON = '/mnt/home/faris/logika-midnight-multiseed-20261030'
KEY = '/home/faris/.ssh/id_ed25519_logika_lab'
os.environ['PATH'] = '/home/faris/.local/bin:' + os.environ['PATH']
parser = argparse.ArgumentParser()
parser.add_argument('--account', type=int, required=True)
N = parser.parse_args().account
P = R / f'private/colab/account{N}'
C = json.loads((R / 'CLOCK.json').read_text())
NAME = f'logika-midnight-a{N}-20261030'
PREFIX = ['/home/faris/.local/bin/colab', '--account', str(N), '--config', str(P / 'sessions.json')]
cache = Path.home() / '.cache/logika-colab-control'
cache.mkdir(parents=True, exist_ok=True)
OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'UserKnownHostsFile=' + str(P / 'known_hosts'), '-o', 'HostKeyAlias=' + NAME,
        '-i', KEY, '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
        '-o', 'ControlPath=' + str(cache / f'midnight-a{N}'), '-o',
        'ProxyCommand=' + shlex.join(PREFIX + ['ssh', '--proxy-mode', '--session', NAME, '--identity', KEY])]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(args, timeout=180):
    remaining=C['deadline_epoch']-time.time()-10
    assert remaining>0, 'Original control deadline reached'
    try:
        return subprocess.run(args,capture_output=True,text=True,check=True,timeout=min(timeout,remaining))
    except subprocess.CalledProcessError as exc:
        (P/'TRANSPORT_FAILURE_PRIVATE.log').write_text(str(exc)+'\n'+(exc.stderr or '')+'\n'+(exc.stdout or ''))
        raise


def ssh(code, timeout=120):
    timeout=min(timeout,max(1,C['deadline_epoch']-time.time()-10))
    assert time.time()<C['deadline_epoch']-10
    try:
        return subprocess.run(['ssh', '-T', *OPTS, 'root@colab', 'python3 -'],
                              input=code, capture_output=True, text=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        (P/'TRANSPORT_FAILURE_PRIVATE.log').write_text(str(exc)+'\n'+(exc.stderr or '')+'\n'+(exc.stdout or ''))
        raise


def remote_json(name):
    return json.loads(ssh('from pathlib import Path;p=Path(' + repr(VROOT + '/' + name) +
                          ');print(p.read_text() if p.exists() else "null")').stdout)


def state(stage, **values):
    dump(P / 'CONTROLLER_STATE.json', dict(epoch=time.time(), account=N, session=NAME,
                                          stage=stage, **values))


def collect():
 code='''from pathlib import Path
import tarfile,json,hashlib
r=Path(ROOT)
with tarfile.open(r/'EVIDENCE.tar.gz','w:gz',compresslevel=1)as a:
 for p in (r/'runs').rglob('*')if (r/'runs').exists()else[]:
  if p.is_file()and p.suffix in ['.json','.npz']:a.add(p,arcname=str(p.relative_to(r)))
 for p in [*r.glob('GATE*.json'),r/'OWNER_FINAL.json',r/'OWNER_STATE.json',r/'ENVIRONMENT_LIVE.json',r/'SETUP_FAILURE.json']:
  if p.is_file():a.add(p,arcname=p.name)
'''.replace('ROOT',repr(VROOT))
 ssh(code);run(['scp','-q',*OPTS,'root@colab:'+VROOT+'/EVIDENCE.tar.gz',str(P/'EVIDENCE.tar.gz')],180);mirror=P/'mirror';mirror.mkdir(exist_ok=True)
 with tarfile.open(P/'EVIDENCE.tar.gz')as a:a.extractall(mirror,filter='data')
 (mirror/'runs').mkdir(exist_ok=True);checked=0
 for p in (mirror/'runs').glob('*/*/RESULT.json'):
  d=json.loads(p.read_text());assert sha(p.parent/'prediction.npz')==d['prediction_sha256']and sha(p.parent/'START.json')==d['start_sha256'];checked+=1
 upload=CANON+f'/private/uploads/account{N}';run(['ssh','-i',KEY,'-o','BatchMode=yes',LAB,'mkdir -p '+upload]);run(['/usr/bin/rsync','-az','-e','ssh -i '+KEY+' -o BatchMode=yes',str(mirror)+'/',LAB+':'+upload+'/'],180);run(['ssh','-i',KEY,'-o','BatchMode=yes',LAB,'python3 '+CANON+'/atomic_collect.py --owner account'+str(N)+' --source '+upload]);dump(P/'COLLECTION_RECEIPT.json',dict(epoch=time.time(),checked_complete_fits=checked));return checked

def main():
 if not (P/'ALLOCATED_READY.json').exists():return
 state('setting_up')
 if not remote_json('BOOTSTRAP_DISPATCH.json'):
  for name in ['CLOCK.json','ENVIRONMENT.json','CHECKPOINTS.json','bootstrap_colab.py']:
   run(['scp','-q',*OPTS,str(R/name),'root@colab:'+VROOT+'/'+name],120)
  code="from pathlib import Path;import subprocess,json,sys;r=Path("+repr(VROOT)+");f=(r/'setup.log').open('ab');p=subprocess.Popen([sys.executable,'-u',str(r/'bootstrap_colab.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);(r/'BOOTSTRAP_DISPATCH.json').write_text(json.dumps({'pid':p.pid}))"
  ssh(code)
 # Dependency setup can run while the sealed scientific payload is being prepared.
 while time.time()<C['admission_end_epoch'] and not (R/'private/PAYLOAD_READY.json').exists():time.sleep(3)
 assert (R/'private/PAYLOAD_READY.json').exists()
 if not remote_json('PAYLOAD_READY.json'):
  run(['scp','-q',*OPTS,str(R/'private/PAYLOAD.tar.gz'),'root@colab:'+VROOT+'/PAYLOAD.tar.gz'],600);run(['ssh','-T',*OPTS,'root@colab','tar -xzf '+VROOT+'/PAYLOAD.tar.gz -C '+VROOT],180);run(['scp','-q',*OPTS,str(P/'OWNER_PLAN.json'),'root@colab:'+VROOT+'/OWNER_PLAN.json'],120);ssh("from pathlib import Path;Path("+repr(VROOT+'/PAYLOAD_READY.json')+").write_text('{}')")
 while time.time()<C['admission_end_epoch']:
  if remote_json('SETUP_FAILURE.json'):raise RuntimeError('Preservedsetupfailure; noautomaticretry')
  if remote_json('ENVIRONMENT_LIVE.json'):break
  time.sleep(10)
 else:raise TimeoutError('Setup exceeded admissionwindow')
 if not remote_json('QUEUE_DISPATCH.json'):
  code="from pathlib import Path;import subprocess,json,os;r=Path("+repr(VROOT)+");f=(r/'queue.log').open('ab');p=subprocess.Popen([str(r/'env311/bin/python'),'-u',str(r/'owner_queue.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','NVIDIA_TF32_OVERRIDE':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8'});(r/'QUEUE_DISPATCH.json').write_text(json.dumps({'pid':p.pid}))"
  ssh(code);dump(P/'TRAINING_STARTED.json',dict(epoch=time.time(),account=N,session=NAME))
 state('training')
 while time.time()<C['deadline_epoch']-120:
  gate=R/'GATE.json'
  if gate.exists():run(['scp','-q',*OPTS,str(gate),'root@colab:'+VROOT+'/GATE.json'],60)
  count=collect();done=remote_json('OWNER_FINAL.json')
  if done:
   dump(P/'FINAL_STATE.json',dict(epoch=time.time(),account=N,session=NAME,completed_fits=count,owner_final=done));break
  time.sleep(30)
 collect();q=run([*PREFIX,'stop','--session',NAME],120);dump(P/'RELEASE_RESPONSE_PRIVATE.json',dict(stdout=q.stdout,stderr=q.stderr));q=run([*PREFIX,'sessions'],60);assert NAME not in q.stdout;dump(P/'RELEASED.json',dict(epoch=time.time(),owned_session=NAME,authenticated_API_absent=True));run(['ssh','-i',KEY,'-o','BatchMode=yes',LAB,"python3 -c "+shlex.quote("from pathlib import Path;import json,time;Path("+repr(CANON+f'/BACKEND_COLLECTION_account{N}_COMPLETE.json')+").write_text(json.dumps({'epoch':time.time(),'owned_released':True}))")]);state('collected_released')
try:main()
except BaseException:
 (P/'CONTROLLER_FAILURE_PRIVATE.log').write_text(traceback.format_exc());state('needs_reconciliation',no_automatic_retry_or_reallocation=True);raise
