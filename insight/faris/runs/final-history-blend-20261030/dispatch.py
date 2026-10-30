"""Bounded20CPUslot execution of the preregistered history blend study."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parent
PLAN = json.loads((R / 'PLAN.json').read_text())
KEY = '/mnt/home/faris/.ssh/id_ed25519_logika_backup119'
SPECS = {
    'cpu2001': dict(ssh=[], root=str(R), python='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python', slots=4),
    'cpu2003': dict(ssh=['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-p2003', 'faris@gpu-labkc.ub.ac.id'], root='/mnt/home/faris/logika-final-history-blend-20261030', python='/mnt/home/faris/logika-submit-best-sub25k-20261029/env_gpu/bin/python', slots=4),
    'cpu119': dict(ssh=['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-i', KEY, 'faris@100.89.211.119'], root='/home/faris/logika-final-history-blend-20261030', python='/home/faris/logika-next-neural-cann-20261029/env311/bin/python', slots=6),
    'cpu42': dict(ssh=['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-i', KEY, 'faris@100.98.169.42'], root='/home/faris/logika-final-history-blend-20261030', python='/home/faris/logika-next-neural-cann-20261029/env311/bin/python', slots=6),
}


def dump(name, value):
    p = R / name
    p.parent.mkdir(exist_ok=True, parents=True)
    q = p.with_name(p.name + '.' + str(os.getpid()) + '.tmp')
    q.write_text(json.dumps(value, indent=2) + '\n')
    q.replace(p)


def run(command, timeout=180):
    remaining = PLAN['deadline_epoch'] - time.time() - 15
    assert remaining > 0
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=min(timeout, remaining))


def install(name, command, working, deadline, cpu):
    folder = Path.home() / '.config/systemd/user'
    folder.mkdir(exist_ok=True, parents=True)
    text = '\n'.join(['[Unit]', 'Description=Owned time-bounded Logika final historical blending', '[Service]',
         'Type=simple', 'ExecStart=' + shlex.join(command), 'WorkingDirectory=' + str(working),
         'UMask=0077', 'Environment=OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=',
         'RuntimeMaxSec=' + str(max(1, int(deadline - time.time() - 15))), 'TimeoutStopSec=10',
         'KillMode=control-group', 'Restart=no', 'MemoryMax=24G', 'CPUQuota=' + str(cpu) + '%',
         'StandardOutput=append:' + str(Path(working) / (name + '.log')),
         'StandardError=append:' + str(Path(working) / (name + '.log')), '[Install]', 'WantedBy=default.target', ''])
    (folder / (name + '.service')).write_text(text)
    run(['systemctl', '--user', 'daemon-reload'])
    run(['systemctl', '--user', 'enable', '--now', name])


def owner(name):
    assert time.time() < PLAN['model_deadline_epoch']
    tasks = json.loads((R / 'ASSIGNMENTS.json').read_text())[name]
    (R / 'logs').mkdir(exist_ok=True)
    def one(task):
        fold, bank, positive = task
        command = [sys.executable, '-u', str(R / 'study.py'), 'worker', '--fold', str(fold), '--bank', bank]
        if positive:
            command.append('--positive')
        label = f'{fold}_{bank}_{positive}'
        with (R / 'logs' / (label + '.log')).open('ab') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=max(1, PLAN['model_deadline_epoch'] - time.time() - 15))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = -1
        return dict(task=task, returncode=code, no_automatic_retry=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=SPECS[name]['slots']) as executor:
        output = list(executor.map(one, tasks))
    dump('OWNER_FINAL_' + name + '.json', dict(owner=name, tasks=output, completed_epoch=time.time(),
                                              fitted_coefficients_not_serialized=True))


def coordinator():
    assert (R / 'PREPUBLICATION.json').is_file()
    run([sys.executable, '-u', str(R / 'study.py'), 'prepare'], timeout=1000)
    tasks = [(f, b, p) for f in range(5) for b in ('all', 'family') for p in (False, True)]
    assignments = {}
    offset = 0
    for name, spec in SPECS.items():
        assignments[name] = tasks[offset:offset + spec['slots']]
        offset += spec['slots']
    assert offset == len(tasks) == 20
    dump('ASSIGNMENTS.json', assignments)
    def dispatch(name):
        spec = SPECS[name]
        if spec['ssh']:
            ssh, target = spec['ssh'], spec['root']
            transfer = ['rsync', '-az', '-e', shlex.join(ssh[:-1])]
            run([*ssh, 'mkdir -p ' + shlex.quote(target + '/bank')])
            files = ['study.py', 'dispatch.py', 'PLAN.json', 'BANK_FROZEN.json', 'ASSIGNMENTS.json', 'PREPUBLICATION.json']
            for filename in files:
                run([*transfer, str(R / filename), ssh[-1] + ':' + target + '/'])
            names = set()
            for fold, bank, positive in assignments[name]:
                names.add(f'meta{fold}.npz')
                names.add(f'fold{fold}.npy' if bank == 'all' else f'family{fold}.npy')
            total = sum((R / 'bank' / n).stat().st_size for n in names)
            probe = 'import shutil;assert shutil.disk_usage(' + repr(target) + ').free>' + str(total + 512 * 1024**2)
            run([*ssh, 'python3 -c ' + shlex.quote(probe)])
            run([*transfer, *[str(R / 'bank' / n) for n in names], ssh[-1] + ':' + target + '/bank/'], timeout=600)
            code = 'import sys;sys.path.insert(0,' + repr(target) + ');from dispatch import install;install(' + repr('logika-finalblend-' + name + '-20261030') + ',' + repr([spec['python'], '-u', target + '/dispatch.py', '--owner', name]) + ',' + repr(target) + ',' + str(PLAN['model_deadline_epoch']) + ',' + str(spec['slots'] * 200) + ')'
            run([*ssh, 'python3 -c ' + shlex.quote(code)])
        else:
            install('logika-finalblend-' + name + '-20261030', [spec['python'], '-u', str(R / 'dispatch.py'), '--owner', name], R, PLAN['model_deadline_epoch'], spec['slots'] * 200)
        dump('DISPATCH_' + name + '.json', dict(epoch=time.time(), owner=name, slots=spec['slots'], tasks=assignments[name]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(dispatch, name): name for name in SPECS}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                dump('DISPATCH_FAILURE_' + futures[future] + '.json', dict(epoch=time.time(), failure='execution_only_dispatch_failure', traceback_private=True))
                (R / 'private' / ('dispatch_' + futures[future] + '.log')).write_text(traceback.format_exc())
    while time.time() < PLAN['deadline_epoch'] - 240:
        for name, spec in SPECS.items():
            if not spec['ssh']:
                continue
            try:
                transfer = ['rsync', '-az', '-e', shlex.join(spec['ssh'][:-1])]
                for child in ('results/',):
                    run([*transfer, spec['ssh'][-1] + ':' + spec['root'] + '/' + child, str(R / child)], timeout=90)
                run([*transfer, '--include=/OWNER_FINAL_*.json', '--exclude=*', spec['ssh'][-1] + ':' + spec['root'] + '/', str(R) + '/'])
            except Exception:
                (R / 'private' / ('collection_' + name + '.log')).write_text(traceback.format_exc())
        done = all((R / ('OWNER_FINAL_' + name + '.json')).exists() for name in SPECS)
        if done or time.time() >= PLAN['model_deadline_epoch'] + 30:
            break
        time.sleep(15)
    run([sys.executable, '-u', str(R / 'study.py'), 'evaluate'], timeout=240)
    dump('COMPLETION.json', dict(epoch=time.time(), status='blends_assessed', all_owners_terminal=done,
                                  deadline_unchanged=True, no_Kaggle_upload=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner', choices=list(SPECS))
    args = parser.parse_args()
    (R / 'private').mkdir(exist_ok=True)
    try:
        owner(args.owner) if args.owner else coordinator()
    except BaseException:
        (R / 'private' / ('failure_dispatch_' + str(os.getpid()) + '.log')).write_text(traceback.format_exc())
        raise
