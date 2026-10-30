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

R = Path('/home/faris/logika-twelvehour-cross-20261030')
VROOT = '/root/logika-twelvehour-cross-20261030'
LAB = 'faris@100.68.29.126'
CANON = '/mnt/home/faris/logika-twelvehour-cross-20261030'
KEY = '/home/faris/.ssh/id_ed25519_logika_lab'
os.environ['PATH'] = '/home/faris/.local/bin:' + os.environ['PATH']
parser = argparse.ArgumentParser()
parser.add_argument('--account', type=int, required=True)
N = parser.parse_args().account
P = R / f'private/colab/account{N}'
C = json.loads((R / 'CLOCK.json').read_text())
NAME = f'logika-cross-a{N}-20261030'
PREFIX = ['/home/faris/.local/bin/colab', '--account', str(N), '--config', str(P / 'sessions.json')]
cache = Path.home() / '.cache/logika-colab-control'
cache.mkdir(parents=True, exist_ok=True)
OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'UserKnownHostsFile=' + str(P / 'known_hosts'), '-o', 'HostKeyAlias=' + NAME,
        '-i', KEY, '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
        '-o', 'ControlPath=' + str(cache / f'cross-a{N}'), '-o',
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
    return subprocess.run(args,capture_output=True,text=True,check=True,timeout=min(timeout,remaining))


def ssh(code, timeout=120):
    timeout=min(timeout,max(1,C['deadline_epoch']-time.time()-10))
    assert time.time()<C['deadline_epoch']-10
    return subprocess.run(['ssh', '-T', *OPTS, 'root@colab', 'python3 -'],
                          input=code, capture_output=True, text=True, check=True, timeout=timeout)


def remote_json(name):
    return json.loads(ssh('from pathlib import Path;p=Path(' + repr(VROOT + '/' + name) +
                          ');print(p.read_text() if p.exists() else "null")').stdout)


def state(stage, **values):
    dump(P / 'CONTROLLER_STATE.json', dict(epoch=time.time(), account=N, session=NAME,
                                          stage=stage, **values))


def collect():
    code = '''from pathlib import Path
import tarfile
r=Path(ROOT)
with tarfile.open(r/'EVIDENCE.tar.gz','w:gz',compresslevel=1) as archive:
 for base in ['runs','logs']:
  if (r/base).exists():
   for path in (r/base).rglob('*'):
    if path.is_file() and path.suffix in ['.json','.npz','.log']:
     archive.add(path,arcname=str(path.relative_to(r)))
 for path in [*r.glob('BACKEND_*.json'),*r.glob('GATE_FAMILY_*.json'),r/'ENVIRONMENT_LIVE.json',r/'SETUP_FAILURE.json']:
  if path.is_file():archive.add(path,arcname=path.name)
'''.replace('ROOT', repr(VROOT))
    ssh(code, timeout=180)
    run(['scp', '-q', *OPTS, 'root@colab:' + VROOT + '/EVIDENCE.tar.gz', str(P / 'EVIDENCE.tar.gz')], 300)
    mirror = P / 'mirror'
    mirror.mkdir(exist_ok=True)
    with tarfile.open(P / 'EVIDENCE.tar.gz') as archive:
        archive.extractall(mirror, filter='data')
    checked = 0
    for path in (mirror / 'runs').glob('*/attempt1/RESULT.json'):
        value = json.loads(path.read_text())
        assert sha(path.parent / 'prediction.npz') == value['prediction_sha256']
        assert sha(path.parent / 'START.json') == value['start_sha256']
        assert value['weights_saved'] is False
        checked += 1
    manifest = {str(path.relative_to(mirror)): dict(bytes=path.stat().st_size, sha256=sha(path))
                for path in mirror.rglob('*') if path.is_file()}
    dump(P / 'PRIVATE_MANIFEST.json', manifest)
    dump(P / 'LOCAL_COLLECTION_RECEIPT.json', dict(epoch=time.time(), checked_fit_records=checked,
          manifest_sha256=sha(P / 'PRIVATE_MANIFEST.json'), private_compact_outputs=True))
    (mirror / 'runs').mkdir(exist_ok=True)
    upload = CANON + f'/private/uploads/account{N}/runs/'
    run(['ssh', '-i', KEY, '-o', 'BatchMode=yes', LAB, 'mkdir -p ' + upload])
    run(['/usr/bin/rsync', '-az', '--ignore-existing', '-e', 'ssh -i ' + KEY + ' -o BatchMode=yes',
         str(mirror / 'runs') + '/', LAB + ':' + upload], 240)
    run(['ssh', '-i', KEY, '-o', 'BatchMode=yes', LAB,
         'python3 ' + CANON + '/ops/atomic_collect.py --owner account' + str(N) +
         ' --source ' + upload], 240)
    metadata = [*mirror.glob('BACKEND_*.json'), *mirror.glob('GATE_FAMILY_*.json'),
                mirror / 'ENVIRONMENT_LIVE.json']
    for path in metadata:
        if not path.exists():
            continue
        name = path.name if path.name.startswith('GATE_FAMILY_') else 'COLAB_A' + str(N) + '_' + path.name
        run(['scp', '-q', '-i', KEY, '-o', 'BatchMode=yes', str(path), LAB + ':' + CANON + '/' + name], 90)
    dump(P / 'CANONICAL_COLLECTION_RECEIPT.json', dict(epoch=time.time(), checked_fit_records=checked,
          manifest_sha256=sha(P / 'PRIVATE_MANIFEST.json')))
    return checked


def setup_or_resume():
    dispatch = remote_json('QUEUE_DISPATCH.json')
    if dispatch is None and (P/'TRAINING_STARTED.json').exists():
        raise RuntimeError('Previously dispatched scientific queue disappeared; reconcile provider replacement before any new training')
    if dispatch is None:
        state('uploading_sealed_source')
        ssh('from pathlib import Path;Path(' + repr(VROOT) + ').mkdir(exist_ok=True)')
        run(['scp', '-q', *OPTS, str(R / 'private/SOURCE.tar.gz'), 'root@colab:' + VROOT + '/SOURCE.tar.gz'], 240)
        run(['ssh', '-T', *OPTS, 'root@colab', 'tar -xzf ' + VROOT + '/SOURCE.tar.gz -C ' + VROOT], 120)
        if remote_json('ENVIRONMENT_LIVE.json') is None and remote_json('BOOTSTRAP_DISPATCH.json') is None:
            code = '''from pathlib import Path
import subprocess,sys,json
r=Path(ROOT)
with (r/'setup.log').open('ab') as stream:
 process=subprocess.Popen([sys.executable,'-u',str(r/'ops/bootstrap_colab.py')],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
(r/'BOOTSTRAP_DISPATCH.json').write_text(json.dumps({'pid':process.pid,'setup_only':True}))
'''.replace('ROOT', repr(VROOT))
            ssh(code)
        if not remote_json('PACKETS_READY.json'):
            plan = json.loads((R / 'EXECUTION_PLAN.json').read_text())['owners'][str(N)]
            designs = json.loads((R / 'CONFIG.json').read_text())['designs']
            wide = any(designs[key]['view'] == 'all' for key in plan['designs'])
            archives=['PACKETS_BASE.tar']+(['PACKETS_WIDE.tar'] if wide else [])
            for name in archives:
                archive=R/'private'/name
                run(['scp','-q',*OPTS,str(archive),'root@colab:'+VROOT+'/'+name],1200)
                run(['ssh','-T',*OPTS,'root@colab','tar -xf '+VROOT+'/'+name+' -C '+VROOT],240)
            ssh('import json;from pathlib import Path;Path(' + repr(VROOT + '/PACKETS_READY.json') + ').write_text(json.dumps({"unpacked":True}))')
        while time.time() < C['admission_end_epoch']:
            if remote_json('SETUP_FAILURE.json'):
                raise RuntimeError('Dependency setup failed; inspect preserved private log')
            if remote_json('ENVIRONMENT_LIVE.json'):
                break
            time.sleep(20)
        else:
            raise TimeoutError('Original admission cutoff reached during setup')
    code = '''from pathlib import Path
import json,subprocess,hashlib,time
r=Path(ROOT);path=r/'QUEUE_DISPATCH.json';reconciled=False
if (r/'BACKEND_FINAL.json').exists():print('already terminal');raise SystemExit
if path.exists():
 previous=json.load(open(path));cmd=Path('/proc')/str(previous['pid'])/'cmdline'
 if cmd.exists() and str(r/'ops/owner_queue.py').encode() in cmd.read_bytes():print('healthy queue retained');raise SystemExit
 for item in json.load(open(r/'DISPATCH.json')).get('dispatches',[]) if (r/'DISPATCH.json').exists() else []:
  worker=Path('/proc')/str(item['pid'])/'cmdline'
  if worker.exists() and str(r/'source/worker.py').encode() in worker.read_bytes() and item['task'].encode() in worker.read_bytes():print('healthy prior owned fits retained; waiting for completion');raise SystemExit
 reconciled=True
 path.rename(r/('QUEUE_DISPATCH.interrupted.'+str(time.time_ns())+'.json'))
assert reconciled or not any((r/'runs').glob('*/attempt1/START.json')), 'Unreconciled prior scientific attempts; no new queue'
inputs=json.load(open(r/'INPUTS.json'));designs=json.load(open(r/'CONFIG.json'))['designs'];owned=json.load(open(r/'EXECUTION_PLAN.json'))['owners'][OWNER]['designs']
for fold in range(5):
 p=r/f'packets/fold{fold}';m=json.load(open(p/'MANIFEST.json'))
 assert hashlib.sha256((p/'MANIFEST.json').read_bytes()).hexdigest()==inputs['packet_manifests'][str(fold)]
 needed={'roles.npz':'roles_sha256','features.parquet':'feature_sha256'}
 if any(designs[key]['view']=='all' for key in owned):needed['all_features.parquet']='full_feature_sha256'
 for name,key in needed.items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==m[key]
(r/'VM_STARTED_TRAINING.json').write_text(json.dumps({'registered_pilots_only_before_gates':True}))
with (r/'backend_queue.log').open('ab') as stream:
 process=subprocess.Popen([str(r/'env311/bin/python'),'-u',str(r/'ops/owner_queue.py'),'--owner',OWNER],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
path.write_text(json.dumps({'pid':process.pid,'owner':OWNER,'original_clock_preserved':True}))
print('owned queue dispatched')
'''.replace('ROOT', repr(VROOT)).replace('OWNER', repr(str(N)))
    ssh(code, timeout=180)
    if not (P/'TRAINING_STARTED.json').exists():dump(P/'TRAINING_STARTED.json',dict(epoch=time.time(),account=N,session=NAME,original_clock_preserved=True))


def main():
    if time.time()>=C['deadline_epoch']-10:
        state('original_deadline_passed_no_new_setup_or_training');return
    if (P / 'RELEASED.json').exists():
        if (P/'FINAL_STATE.json').exists() and (P/'CANONICAL_COLLECTION_RECEIPT.json').exists():
            run(['scp','-q','-i',KEY,'-o','BatchMode=yes',str(P/'FINAL_STATE.json'),LAB+':'+CANON+'/BACKEND_COLLECTION_account'+str(N)+'_COMPLETE.json'])
        return
    assert json.loads((P / 'ALLOCATION_RESPONSE.json').read_text())['returncode'] == 0
    publication = json.loads((R / 'PREPUBLICATION.json').read_text())
    assert publication['config_sha256'] == sha(R / 'CONFIG.json')
    assert publication['source_seal_sha256'] == sha(R / 'SOURCE_SEAL.json')
    normal = False
    try:
        if time.time() < C['model_deadline_epoch']:
            setup_or_resume()
        while time.time() < C['model_deadline_epoch'] + 30:
            try:
                if not remote_json('BACKEND_FINAL.json'):setup_or_resume()
                checked = collect()
                final = remote_json('BACKEND_FINAL.json')
                state('running' if not final else 'terminal_collected', checked_fit_records=checked)
                if final:
                    normal = True
                    break
            except Exception:
                (P / 'COLLECTION_FAILURE_PRIVATE.log').write_text(traceback.format_exc())
                state('collection_retry_preserving_healthy_queue')
            time.sleep(30)
        if not normal and time.time() >= C['model_deadline_epoch']:
            normal = True
            try:
                collect()
            except Exception:
                (P / 'FINAL_COLLECTION_FAILURE_PRIVATE.log').write_text(traceback.format_exc())
    except Exception:
        (P / 'CONTROLLER_FAILURE_PRIVATE.log').write_text(traceback.format_exc())
        state('execution_review_required_no_implicit_retraining')
        if time.time() < C['model_deadline_epoch']:
            raise
        normal = True
    if normal:
        response = subprocess.run([*PREFIX, 'stop', '--session', NAME], capture_output=True, text=True, timeout=150)
        dump(P / 'RELEASE_RESPONSE.json', dict(epoch=time.time(), returncode=response.returncode,
             stdout=response.stdout, stderr=response.stderr))
        listing = subprocess.run([*PREFIX, 'sessions'], capture_output=True, text=True, timeout=90)
        dump(P / 'POST_RELEASE_SESSIONS.json', dict(epoch=time.time(), returncode=listing.returncode, stdout=listing.stdout, stderr=listing.stderr))
        released = response.returncode == 0 and listing.returncode == 0 and NAME not in listing.stdout
        if released:
            dump(P / 'RELEASED.json', dict(epoch=time.time(), session=NAME, owned_only=True))
        result = dict(epoch=time.time(), account=N, session=NAME, released=released,
                      canonical_collected=(P / 'CANONICAL_COLLECTION_RECEIPT.json').exists(),
                      local_hash_verified=(P / 'LOCAL_COLLECTION_RECEIPT.json').exists(),
                      original_deadline_epoch=C['deadline_epoch'])
        dump(P / 'FINAL_STATE.json', result)
        if released and result['canonical_collected']:
            run(['scp', '-q', '-i', KEY, '-o', 'BatchMode=yes', str(P / 'FINAL_STATE.json'),
                 LAB + ':' + CANON + '/BACKEND_COLLECTION_account' + str(N) + '_COMPLETE.json'])


if __name__ == '__main__':
    main()
