"""Admit, run and collect one execution-only spillover within the immutable clock.

Training and evaluation stay in the sealed original scientific modules. This
controller has no model-selection logic and never reads candidate outer scores.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import traceback
from colab_transport import LOCAL, PRIVATE, REMOTE, PREFIX, SESSION, ssh, put, get

LAB = '/mnt/home/faris/logika-feature-subspace-moe-20261028'
PYTHON = '/mnt/home/faris/logika-model-arena-20261025/env311/bin/python'
TASK = 'kumo__expanded_cashflow__s2026__f0'
CLOCK_SHA = '693c430a2cdfe6cfc6c540e7e1e9480a0ae8944763b4cb1598da90b673456bac'
CLOCK = json.loads((LOCAL / 'CLOCK.json').read_text())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def write(value, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def lab(code, timeout=90):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                           'faris@100.68.29.126', PYTHON + ' -c ' + shlex.quote(code)],
                          text=True, capture_output=True, timeout=timeout, check=True)

def remote(code, timeout=90):
    return ssh('python3 -', code=code, timeout=timeout).stdout.strip()

def colab_json(relative):
    return json.loads(remote("from pathlib import Path;import json;p=Path(" + repr(REMOTE + '/' + relative)
                             + ");print(p.read_text()if p.exists()else'null')"))

def note(stage, **fields):
    value = dict(stage=stage, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 original_clock_sha256=CLOCK_SHA, **fields)
    write(value, PRIVATE / 'CONTROLLER_STATE.json')
    print(json.dumps(value), flush=True)

def initialize():
    assert digest(LOCAL / 'CLOCK.json') == CLOCK_SHA
    assert time.time() < CLOCK['admission_end_epoch']
    for name in ['colab_execution_adapter.py', 'colab_batch.py']:
        put(LOCAL / 'ops' / name, REMOTE + '/ops/' + name)
    for name in ['EXECUTION_SPILLOVER_PLAN.json', 'COLAB_EXECUTION_ADDENDUM.json']:
        subprocess.run(['scp', '-q', str(LOCAL / name), 'faris@100.68.29.126:' + LAB + '/' + name], check=True, timeout=90)
    subprocess.run(['scp', '-q', str(LOCAL / 'ops/spillover_cutover.py'),
                    'faris@100.68.29.126:' + LAB + '/ops/spillover_cutover.py'], check=True, timeout=90)

def replay():
    if (PRIVATE / 'COLAB_REPLAY_GATE.json').exists():
        result = json.loads((PRIVATE / 'COLAB_REPLAY_GATE.json').read_text())
        assert result['status'] == 'PASS'
        return result
    note('waiting_for_verified_fold0')
    while time.time() < CLOCK['admission_end_epoch'] - 1800:
        environment = colab_json('COLAB_ENVIRONMENT.json')
        assert not (PRIVATE / 'STREAM_FAILURE.json').exists()
        if environment is not None:
            write(environment, LOCAL / 'COLAB_ENVIRONMENT.json')
            ready = remote("""from pathlib import Path
import sys,json
r=Path.home()/'logika-feature-subspace-moe-20261028';sys.path.insert(0,str(r/'source'))
from common import check_seal,sha,load
check_seal();p=r/'packets/fold0';ready=False
if(p/'roles.npz').exists():
 m=json.loads((p/'MANIFEST.json').read_text())
 ready=sha(p/'MANIFEST.json')==load('GATE.json')['packet_manifests']['0'] and sha(p/'features.parquet')==m['feature_sha256'] and sha(p/'roles.npz')==m['roles_sha256']
print(json.dumps({'ready':ready}))
""")
            if json.loads(ready)['ready']:
                break
        time.sleep(30)
    else:
        raise RuntimeError('Original admission time exhausted during transfer')
    reference = PRIVATE / 'lab_reference.npz'
    if not reference.exists():
        subprocess.run(['scp', '-q', 'faris@100.68.29.126:' + LAB + '/runs/' + TASK + '/attempt1/prediction.npz', str(reference)], check=True, timeout=120)
    put(reference, REMOTE + '/lab_reference.npz')
    note('label_free_replay', task=TASK)
    remote("""from pathlib import Path
import subprocess,json,time,os
r=Path.home()/'logika-feature-subspace-moe-20261028';task=TASK
if not(r/f'runs/{task}/attempt2').exists()and not(r/'REPLAY_DISPATCH.json').exists():
 with(r/'replay.log').open('ab')as f:
  p=subprocess.Popen([str(r/'env311/bin/python'),str(r/'ops/colab_execution_adapter.py'),'--task',task,'--attempt','2'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
 (r/'REPLAY_DISPATCH.json').write_text(json.dumps({'pid':p.pid,'epoch':time.time()})+'\\n')
print('Replay dispatched or reconciled')
""".replace('TASK', repr(TASK)))
    while time.time() < CLOCK['admission_end_epoch'] - 600:
        record = colab_json(f'runs/{TASK}/attempt2/RESULT.json')
        failure = colab_json(f'runs/{TASK}/attempt2/FAILURE.json')
        if failure:
            write(failure, PRIVATE / 'REPLAY_PRIVATE_FAILURE.json')
            raise RuntimeError('Replay execution failed; private evidence preserved')
        if record:
            break
        dispatch = colab_json('REPLAY_DISPATCH.json')
        assert time.time() - dispatch['epoch'] < 1850, 'Replay genuine time limit exceeded'
        time.sleep(20)
    else:
        raise RuntimeError('Replay did not finish before original admission reserve')
    code = r'''
from pathlib import Path
import sys,json,numpy as np
r=Path.home()/'logika-feature-subspace-moe-20261028';sys.path.insert(0,str(r/'source'))
from common import sha,dump,check_seal
check_seal();a=np.load(r/'lab_reference.npz');b=np.load(r/'runs/TASK/attempt2/prediction.npz')
assert np.array_equal(a['rows'],b['rows'])
d=np.abs(a['raw']-b['raw']);cor=float(np.corrcoef(a['raw'],b['raw'])[0,1])
result={'status':'PASS'if float(d.mean())<=.001 and float(d.max())<=.02 and cor>=.999 else'FAIL','mean_abs_probability_difference':float(d.mean()),'max_abs_probability_difference':float(d.max()),'correlation':cor,'rows_exact':True,'rows':len(d),'reference_prediction_sha256':sha(r/'lab_reference.npz'),'replay_prediction_sha256':sha(r/'runs/TASK/attempt2/prediction.npz'),'adapter_sha256':sha(r/'ops/colab_execution_adapter.py'),'checkpoint_sha256':'cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613','original_clock_sha256':sha(r/'CLOCK.json'),'labels_used_for_acceptance':False,'GPU_bitwise_equivalence_claimed':False}
dump(result,r/'COLAB_REPLAY_GATE.json');print(json.dumps(result))
'''.replace('TASK', TASK)
    result = json.loads(ssh(REMOTE + '/env311/bin/python -', code=code, timeout=90).stdout)
    write(result, PRIVATE / 'COLAB_REPLAY_GATE.json')
    write(result, LOCAL / 'COLAB_REPLAY_GATE.json')
    assert result['status'] == 'PASS', 'Label-free replay tolerance failed; lab cohort stays intact'
    for name in ['COLAB_REPLAY_GATE.json', 'COLAB_ENVIRONMENT.json']:
        subprocess.run(['scp', '-q', str(LOCAL / name), 'faris@100.68.29.126:' + LAB + '/' + name], check=True, timeout=90)
    return result

def cutover():
    note('reconciling_owned_lab_jobs')
    result = lab("import runpy;runpy.run_path(" + repr(LAB + '/ops/spillover_cutover.py') + ",run_name='__main__')", timeout=2100)
    (PRIVATE / 'cutover_receipt.log').write_text(result.stdout)
    note('A100_cohort_admitted', target_Kumo_fits=80, CPU_fits_retained=180)

def collect_new(collected):
    code = """from pathlib import Path
import json,tarfile,hashlib
r=Path.home()/'logika-feature-subspace-moe-20261028';previous=set(PREVIOUS)
tasks=sorted({p.parent.parent.name for p in(r/'runs').glob('*/attempt2/RESULT.json')})
new=[task for task in tasks if task not in previous]
if new:
 with tarfile.open(r/'RESULT_RETURN_PRIVATE.tar.gz','w:gz')as tar:
  for task in new:tar.add(r/f'runs/{task}/attempt2',arcname=f'runs/{task}/attempt2')
 h=hashlib.sha256((r/'RESULT_RETURN_PRIVATE.tar.gz').read_bytes()).hexdigest()
else:h=None
print(json.dumps({'new':new,'sha256':h}))
""".replace('PREVIOUS', repr(sorted(collected)))
    receipt = json.loads(remote(code))
    if not receipt['new']:
        return collected
    archive = PRIVATE / 'returns' / ('return_' + str(len(collected)) + '.tar.gz')
    archive.parent.mkdir(exist_ok=True)
    get(REMOTE + '/RESULT_RETURN_PRIVATE.tar.gz', archive)
    assert digest(archive) == receipt['sha256']
    target = PRIVATE / 'persistent_results'
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            assert member.name.startswith('runs/kumo__') and '..' not in Path(member.name).parts
            assert not member.issym() and not member.islnk()
        tar.extractall(target, filter='data')
    subprocess.run(['rsync', '-az', str(target / 'runs') + '/',
                    'faris@100.68.29.126:' + LAB + '/runs/'], check=True, timeout=180)
    collected |= set(receipt['new'])
    write(sorted(collected), PRIVATE / 'COLLECTED_TASKS.json')
    return collected

def batch():
    remote("""from pathlib import Path
import subprocess,json,time
r=Path.home()/'logika-feature-subspace-moe-20261028'
if not(r/'BATCH_DISPATCH.json').exists():
 with(r/'batch.log').open('ab')as f:
  p=subprocess.Popen([str(r/'env311/bin/python'),str(r/'ops/colab_batch.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (r/'BATCH_DISPATCH.json').write_text(json.dumps({'pid':p.pid,'epoch':time.time()})+'\\n')
print('Batch dispatched or reconciled')
""")
    collected = set(json.loads((PRIVATE / 'COLLECTED_TASKS.json').read_text())) if (PRIVATE / 'COLLECTED_TASKS.json').exists() else set()
    while time.time() < CLOCK['deadline_epoch'] - 7100:
        collected = collect_new(collected)
        state = colab_json('COLAB_LIVE_STATE.json') or {}
        terminal = colab_json('COLAB_BATCH_RESULT.json')
        failure = colab_json('COLAB_BATCH_FAILURE.json')
        if failure:
            write(failure, PRIVATE / 'BATCH_PRIVATE_FAILURE.json')
            raise RuntimeError('A100 batch failed; preserve attempts and inspect')
        state.update(stage='modelling', backend='Colab account2 A100', completed=180 + len(collected),
                     expected=260, pending=80 - len(collected), clock=CLOCK,
                     UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     original_clock_sha256=CLOCK_SHA, canonical_collected_Kumo_fits=len(collected))
        # This aggregate state carries no labels or candidate scores.
        lab("from pathlib import Path;import json;p=Path(" + repr(LAB + '/LIVE_STATE.json') + ");p.write_text(" + repr(json.dumps(state, indent=2) + '\n') + ")")
        note('A100_modelling', collected_Kumo_fits=len(collected), expected_Kumo_fits=80,
             active=state.get('active'))
        if terminal:
            write(terminal, LOCAL / 'COLAB_BATCH_RESULT.json')
            collected = collect_new(collected)
            assert terminal['completed'] == len(collected)
            break
        time.sleep(45)
    else:
        raise RuntimeError('Original computation reserve reached; inspect A100 process')
    assert len(collected) == 80, 'Partial Kumo cohort is not a complete experiment'
    assert (PRIVATE / 'STREAM_COMPLETE.json').exists()
    manifest = json.loads((PRIVATE / 'input_bundle.json').read_text())
    check = remote("from pathlib import Path;import hashlib;print(hashlib.sha256(Path(" + repr(REMOTE + '/COLAB_INPUTS_PRIVATE.tar.gz') + ").read_bytes()).hexdigest())")
    assert check == manifest['sha256']
    # Verify the canonical cohort before releasing its ephemeral runtime.
    audit = lab("""from pathlib import Path
import sys,json
r=Path(ROOT);sys.path.insert(0,str(r/'source'))
from common import check_seal,load,sha,dump
from compose import completed
check_seal();tasks=load('CONFIG.json')['tasks'];paths={k:completed(k)for k in tasks}
assert all(p is not None for p in paths.values())
for key,p in paths.items():
 if tasks[key]['family']=='kumo':
  assert p.name=='attempt2' and(p/'EXECUTION_BACKEND.json').exists()
record={'status':'PASS','expert_fits':len(paths),'Kumo_backend_cohort':'Colab account2 A100; all80 fits','original_clock_sha256':sha(r/'CLOCK.json'),'prediction_hashes':{key:sha(p/'prediction.npz')for key,p in paths.items()}}
dump(record,r/'SPILLOVER_COLLECTION_AUDIT.json');print(json.dumps(record))
""".replace('ROOT', repr(LAB)))
    write(json.loads(audit.stdout), LOCAL / 'SPILLOVER_COLLECTION_AUDIT.json')
    for name in ['COLAB_BATCH_RESULT.json', 'COLAB_ENVIRONMENT.json', 'COLAB_REPLAY_GATE.json', 'RELEASE_WATCHDOG.json']:
        record = colab_json(name)
        if record is not None:
            write(record, PRIVATE / name)
    get(REMOTE + '/bootstrap.log', PRIVATE / 'bootstrap_complete.log')
    subprocess.run([*PREFIX, 'stop', '--session', SESSION], check=True, timeout=120,
                   stdout=(PRIVATE / 'runtime_stop.log').open('a'), stderr=subprocess.STDOUT)
    write({'status': 'stopped_after_verified_collection', 'epoch': time.time(), 'original_clock_sha256': CLOCK_SHA}, LOCAL / 'COLAB_RELEASE_RECEIPT.json')
    note('collected_and_runtime_released', completed=260)

def resume_original_evaluation():
    assert time.time() < CLOCK['deadline_epoch'] - 5400
    for name in ['COLAB_BATCH_RESULT.json', 'COLAB_RELEASE_RECEIPT.json']:
        subprocess.run(['scp', '-q', str(LOCAL / name), 'faris@100.68.29.126:' + LAB + '/' + name], check=True, timeout=90)
    code = """from pathlib import Path
import sys,time,subprocess,json
r=Path(ROOT);sys.path.insert(0,str(r/'source'))
from common import check_seal,dump,sha
clock=check_seal();remaining=int(clock['deadline_epoch']-time.time());assert remaining>5400
unit='logika-moe28-postspillover-resume'
cmd=['systemd-run','--user','--quiet','--unit='+unit,'-p','WorkingDirectory='+str(r),'-p','MemoryMax=8G','-p','MemorySwapMax=0','-p','RuntimeMaxSec='+str(remaining),sys.executable,str(r/'source/coordinator.py'),'--resume']
subprocess.run(cmd,check=True)
dump({'unit':unit+'.service','runtime_limit_seconds':remaining,'clock_sha256':sha(r/'CLOCK.json'),'new_clock_created':False},r/'SPILLOVER_RESUME_RECEIPT.json')
print(unit+'.service')
""".replace('ROOT', repr(LAB))
    print(lab(code).stdout, flush=True)
    note('original_sealed_evaluation_resumed', completed=260, scientific_selection_changed=False)

def main():
    lock = (PRIVATE / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initialize()
    replay()
    cutover()
    batch()
    resume_original_evaluation()

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write({'error': traceback.format_exc(), 'epoch': time.time(), 'original_clock_sha256': CLOCK_SHA}, PRIVATE / 'CONTROLLER_FAILURE.json')
        raise
