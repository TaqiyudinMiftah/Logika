"""Reuse exactly the20idleCPUslots and immutable prediction bank."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import concurrent.futures
import json
import shlex
import subprocess
import sys
import time
from dispatch import R, SPECS, run, install, dump


def owner(name):
    plan = json.loads((R / 'GREEDY_PLAN.json').read_text())
    tasks = json.loads((R / 'ASSIGNMENTS.json').read_text())[name]
    (R / 'logs').mkdir(exist_ok=True)
    def one(task):
        fold, bank, ks = task
        command = [sys.executable, '-u', str(R / 'greedy.py'), '--fold', str(fold), '--bank', bank] + (['--ks'] if ks else [])
        with (R / 'logs' / f'greedy_{fold}_{bank}_{ks}.log').open('ab') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=max(1, plan['model_deadline_epoch'] - time.time() - 15))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -1
        return dict(task=task, returncode=code, no_automatic_retry=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=SPECS[name]['slots']) as executor:
        result = list(executor.map(one, tasks))
    dump('GREEDY_OWNER_FINAL_' + name + '.json', dict(owner=name, tasks=result, epoch=time.time()))


def deploy():
    plan = json.loads((R / 'GREEDY_PLAN.json').read_text())
    for name, spec in SPECS.items():
        target = spec['root']
        if spec['ssh']:
            ssh = spec['ssh']
            transfer = ['rsync', '-az', '-e', shlex.join(ssh[:-1])]
            files = ['greedy.py', 'greedy_dispatch.py', 'GREEDY_PLAN.json', 'GREEDY_PREPUBLICATION.json']
            run([*transfer, *[str(R / n) for n in files], ssh[-1] + ':' + target + '/'])
            code = 'import sys;sys.path.insert(0,' + repr(target) + ');from dispatch import install;install(' + repr('logika-finalgreedy-' + name + '-20261030') + ',' + repr([spec['python'], '-u', target + '/greedy_dispatch.py', '--owner', name]) + ',' + repr(target) + ',' + str(plan['model_deadline_epoch']) + ',' + str(spec['slots'] * 200) + ')'
            run([*ssh, 'python3 -c ' + shlex.quote(code)])
        else:
            install('logika-finalgreedy-' + name + '-20261030', [spec['python'], '-u', str(R / 'greedy_dispatch.py'), '--owner', name], R, plan['model_deadline_epoch'], spec['slots'] * 200)
        print(name, 'dispatched retained bank and useful heldmeta ensemble selection', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--owner', choices=list(SPECS))
    a = p.parse_args()
    owner(a.owner) if a.owner else deploy()
