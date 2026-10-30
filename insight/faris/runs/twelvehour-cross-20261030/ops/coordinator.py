"""Persistent collection and owned-lab dispatch, isolated from long assessments."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import fcntl
import json
import shlex
import shutil
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
sys.path.insert(0, str(R / 'ops'))
from common import load, dump, sha, utc, check_seal
from atomic_collect import publish
from service_install import install

EVAL_PY = '/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
KEY = '/mnt/home/faris/.ssh/id_ed25519_logika_backup119'
CPU_PY = '/mnt/home/faris/logika-model-arena-20261025/env311/bin/python'
GPU_PY = '/mnt/home/faris/logika-submit-best-sub25k-20261029/env_gpu/bin/python'
PERSONAL_PY = '/home/faris/logika-next-neural-cann-20261029/env311/bin/python'
SPECS = {
    'cpu2001': dict(ssh=[], root=str(R / 'private/backends/cpu2001'), python=CPU_PY,
                    packets='/mnt/home/faris/logika-twelvehour-next-20261029/packets', memory='28G'),
    'cpu2003': dict(ssh=['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-p2003', 'faris@gpu-labkc.ub.ac.id'],
                    root='/mnt/home/faris/logika-cross-20261030/owner_cpu2003', python=GPU_PY,
                    packets='/mnt/home/faris/logika-twelvehour-20261029/packets', memory='28G'),
    'gpu2003': dict(ssh=['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-p2003', 'faris@gpu-labkc.ub.ac.id'],
                    root='/mnt/home/faris/logika-cross-20261030/owner_gpu2003', python=GPU_PY,
                    packets='/mnt/home/faris/logika-twelvehour-20261029/packets', memory='22G'),
    'cpu119': dict(ssh=['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-i', KEY, 'faris@100.89.211.119'],
                   root='/home/faris/logika-cross-20261030/owner_cpu119', python=PERSONAL_PY,
                   packets='/home/faris/logika-twelvehour-next-20261029/private/shared-packets/packets', memory='40G'),
    'cpu42': dict(ssh=['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-i', KEY, 'faris@100.98.169.42'],
                  root='/home/faris/logika-cross-20261030/owner_cpu42', python=PERSONAL_PY,
                  packets='/home/faris/logika-next-cpu-exactks-20261029/shared-packets/packets', memory='40G'),
}
FILES = ['CONFIG.json', 'SOURCE_SEAL.json', 'VALIDATION_CONTRACT.json', 'INPUTS.json',
         'ENVIRONMENT.json', 'CHECKPOINTS.json', 'EXECUTION_PLAN.json', 'DISPATCH_SEAL.json',
         'CLOCK.json', 'SYNTHETIC_CHECKS.json', 'GATE.json', 'TRAINING_BRIEF.md', 'DEADLINE.md', 'PREPUBLICATION.json',
         'EXTRA_DESIGNS.jsonl', 'BASELINE_REPLAY_GATE.json', 'REFERENCE_SEED_PROVENANCE.json', 'PLAN.json', 'README.md']


def call(args, timeout=120):
    remaining=load('CLOCK.json')['deadline_epoch']-time.time()-15
    assert remaining>0, 'Original coordination deadline reached'
    timeout=min(timeout,remaining)
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=timeout)


def setup(owner, start=True):
    spec = SPECS[owner]
    assignment = load('EXECUTION_PLAN.json')['owners'][owner]
    destination, ssh = Path(spec['root']), spec['ssh']
    if ssh:
        call([*ssh, 'mkdir -p ' + shlex.quote(str(destination)) + ' && chmod 700 ' + shlex.quote(str(destination))])
        transport = shlex.join(ssh[:-1])
        for name in [*FILES, 'source', 'ops', 'evidence']:
            if not (R/name).exists():continue
            call(['rsync', '-az', '-e', transport, str(R / name), ssh[-1] + ':' + str(destination) + '/'], 180)
        code = ("from pathlib import Path;import hashlib,json; r=Path(" + repr(str(destination)) + ");"
                "p=Path(" + repr(spec['packets']) + "); assert p.is_dir();"
                "(r/'packets').symlink_to(p,target_is_directory=True) if not(r/'packets').exists() else None;"
                "[(r/n).mkdir(exist_ok=True)for n in ['private','logs','runs']];"
                "inputs=json.load(open(r/'INPUTS.json'));"
                "assert all(hashlib.sha256((p/f'fold{f}/MANIFEST.json').read_bytes()).hexdigest()==h for f,h in inputs['packet_manifests'].items())")
        call([*ssh, 'python3 -c ' + shlex.quote(code)])
    else:
        destination.mkdir(parents=True, exist_ok=True);destination.chmod(0o700)
        for name in FILES:
            if not (R/name).exists():continue
            shutil.copy2(R / name, destination / name)
        for name in ['source', 'ops', 'evidence']:
            shutil.copytree(R / name, destination / name, dirs_exist_ok=True)
        if not (destination / 'packets').exists():
            (destination / 'packets').symlink_to(spec['packets'], target_is_directory=True)
        for name in ['private', 'logs', 'runs']:
            (destination / name).mkdir(exist_ok=True)
    needs_wide = any(load('CONFIG.json')['designs'][key]['view'] == 'all'
                     for key in assignment['designs'])
    if ssh and needs_wide:
        code = ("from pathlib import Path;r=Path(" + repr(str(destination)) + ");p=Path(" + repr(spec['packets']) + ");"
                "q=r/'packets';q.unlink() if q.is_symlink() else None;q.mkdir(exist_ok=True);"
                "[(q/f'fold{f}').mkdir(exist_ok=True)for f in range(5)];"
                "[(q/f'fold{f}'/n).symlink_to(p/f'fold{f}'/n)for f in range(5)for n in "
                "['features.parquet','roles.npz','inner_roles.npz','anchor.npz','MANIFEST.json']if not(q/f'fold{f}'/n).exists()]")
        call([*ssh, 'python3 -c ' + shlex.quote(code)])
        for fold in range(5):
            call(['rsync', '-az', '--partial', '-e', shlex.join(ssh[:-1]),
                  str(R / f'packets/fold{fold}/all_features.parquet'),
                  ssh[-1] + ':' + str(destination / f'packets/fold{fold}/')], 240)
    probe = ("import json,importlib.metadata;from pathlib import Path;r=Path(" + repr(str(destination)) + ");"
             "expected=json.load(open(r/'ENVIRONMENT.json'))['required_versions'];"
             "actual={n:importlib.metadata.version(n)for n in expected};"
             "assert all(actual[n].split('+')[0]==v for n,v in expected.items()),repr(actual);"
             "print(json.dumps({'status':'PASS','versions':actual,'fit_started':False}))")
    result = call([*ssh, shlex.join([spec['python'], '-c', probe])] if ssh else
                  [spec['python'], '-c', probe])
    dump(json.loads(result.stdout), R / ('ENVIRONMENT_PREFLIGHT_' + owner + '.json'))
    name = 'logika-cross-' + owner + '-20261030'
    arguments = [spec['python'], '-u', str(destination / 'ops/owner_queue.py'), '--owner', owner]
    code = ("import sys;from pathlib import Path;sys.path.insert(0," + repr(str(destination / 'ops')) + ");"
            "from service_install import install;install(" + repr(name) + ',' + repr(arguments) + ',' +
            repr(str(destination)) + ',memory=' + repr(spec['memory']) + ',cpu=' + str(assignment['slots'] * 200) + ',start=' + repr(start) + ')')
    if ssh:
        call([*ssh, 'python3 -c ' + shlex.quote(code)], 90)
    else:
        install(name, arguments, destination, memory=spec['memory'], cpu=assignment['slots'] * 200, start=start)
    if not start:return
    dump(dict(UTC=utc(), owner=owner, root=str(destination), unit=name,
              persistent_enabled_user_service=True, clock_sha256=sha(R / 'CLOCK.json')),
         R / ('DISPATCH_' + owner + '.json'))


def collect(owner):
    spec, ssh = SPECS[owner], SPECS[owner]['ssh']
    source = spec['root'] + '/runs'
    if ssh:
        source = ssh[-1] + ':' + source
        os.environ['LOGIKA_COLLECT_RSH'] = shlex.join(ssh[:-1])
    try:
        receipt = publish(source, owner)
    finally:
        os.environ.pop('LOGIKA_COLLECT_RSH', None)
    metadata = R / 'private/owner_metadata' / owner
    metadata.mkdir(parents=True, exist_ok=True)
    if ssh:
        call(['rsync', '-az', '-e', shlex.join(ssh[:-1]), '--include=/BACKEND_*.json', '--include=/GATE_FAMILY_*.json',
              '--include=/OWNER_*.json', '--exclude=*', ssh[-1] + ':' + spec['root'] + '/', str(metadata) + '/'], 60)
    else:
        for path in Path(spec['root']).glob('BACKEND_*.json'):
            shutil.copy2(path, metadata / path.name)
        for path in Path(spec['root']).glob('GATE_FAMILY_*.json'):
            shutil.copy2(path, metadata / path.name)
    for path in metadata.glob('GATE_FAMILY_*.json'):
        target = R / path.name
        if target.exists():
            assert sha(target) == sha(path), 'A runtime qualification changed after publication'
        else:
            shutil.copy2(path, target)
    if (metadata / 'BACKEND_FINAL.json').exists():
        dump(dict(UTC=utc(), owner=owner, receipt=receipt,
                  terminal_sha256=sha(metadata / 'BACKEND_FINAL.json')),
             R / ('COLLECTED_' + owner + '.json'))


def main():
    for name in ['private', 'logs', 'runs', 'evaluation_inputs']:
        (R / name).mkdir(exist_ok=True)
    lock = (R / 'coordinator.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    clock = check_seal()
    assert load('GATE.json')['status'] == 'PASS' and load('BASELINE_REPLAY_GATE.json')['status']=='PASS'
    assert load('DISPATCH_SEAL.json')['clock_sha256'] == sha(R / 'CLOCK.json')
    plan = load('EXECUTION_PLAN.json')['owners']
    assert load('DISPATCH_SEAL.json')['execution_plan_sha256'] == sha(R / 'EXECUTION_PLAN.json')
    for mode in ['compose', 'evaluate']:
        install('logika-cross-' + mode + '-20261030',
                [EVAL_PY, '-u', str(R / 'ops/assessment_loop.py'), '--mode', mode],
                R, memory='16G', cpu=200)
    while time.time() < clock['deadline_epoch'] - 90:
        errors = {}
        for owner in [key for key in plan if key in SPECS]:
            try:
                if not (R / ('DISPATCH_' + owner + '.json')).exists() and time.time() < clock['admission_end_epoch']:
                    setup(owner)
                if (R / ('DISPATCH_' + owner + '.json')).exists():
                    collect(owner)
            except Exception as exc:
                errors[owner] = type(exc).__name__
                (R / 'private' / ('collection_' + owner + '.log')).write_text(traceback.format_exc())
        owners_done = all((R / ('COLLECTED_' + owner + '.json')).exists() if owner in SPECS
                          else (R / ('BACKEND_COLLECTION_account' + owner + '_COMPLETE.json')).exists()
                          for owner in plan)
        if owners_done or time.time() >= clock['model_deadline_epoch'] + 90:
            if not (R / 'COMPUTATION_FINAL.json').exists():
                dump(dict(UTC=utc(), all_owner_collections_verified=owners_done,
                          unavailable_owners=[owner for owner in plan if not (
                              (R / ('COLLECTED_' + owner + '.json')).exists() if owner in SPECS
                              else (R / ('BACKEND_COLLECTION_account' + owner + '_COMPLETE.json')).exists())],
                          no_unknown_attempts_retried=True), R / 'COMPUTATION_FINAL.json')
        result = dict(stage='complete' if (R / 'COMPLETION.json').exists() else 'running', UTC=utc(),
                      completed_fits=len(list((R / 'runs').glob('*/attempt1/RESULT.json'))),
                      failure_records=len(list((R / 'runs').glob('*/attempt1/FAILURE.json'))),
                      interruptions=len(list((R / 'runs').glob('*/attempt1/INTERRUPTION.json'))),
                      planned_fits=len(load('CONFIG.json')['designs']) * 10,
                      owner_collections_complete=owners_done, errors=errors,
                      clock_sha256=sha(R / 'CLOCK.json'), no_automatic_submission=True)
        dump(result, R / 'LIVE_STATE.json')
        dump(result, R / 'RESUME_STATE.json')
        if (R / 'COMPLETION.json').exists():
            return
        time.sleep(20)
    dump(dict(stage='original_deadline_reached', UTC=utc(),
              completed=(R / 'COMPLETION.json').exists(), no_clock_extension=True), R / 'RESUME_STATE.json')


if __name__ == '__main__':
    main()
