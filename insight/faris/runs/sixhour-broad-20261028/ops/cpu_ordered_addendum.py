"""Runtime-gated, append-only CPU comparison under the original campaign clock.

No scientific source, features, evaluator, selection rule or fitted state changes.
Only explicitly preregistered recipes use the existing worker and its boundaries.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import check_seal, dump, load, sha, utc

C = check_seal()
PLAN = load('CPU_ORDERED_PLAN.json')
HOST = 'faris@gpu-labkc.ub.ac.id'
SSH = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=15', '-p2003', HOST]
TRANSPORT = 'ssh -oBatchMode=yes -oConnectTimeout=15 -p2003'
PYTHON = str(R / 'env311/bin/python')


def run(args, timeout=180, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, check=True,
                          timeout=timeout, **kwargs)


def retire_flag(name):
    path = R / name
    if path.exists():
        directory = R / 'execution_flag_history'
        directory.mkdir(exist_ok=True)
        target = directory / (path.stem + '.' + sha(path)[:16] + path.suffix)
        if not target.exists():
            target.write_bytes(path.read_bytes())
        path.unlink()


def pilot(task):
    folder = R / 'runs' / task / 'attempt1'
    if not (folder / 'RESULT.json').exists():
        assert not folder.exists(), 'Existing failed/interrupted attempt requires explicit reconciliation'
        with (R / 'logs' / ('ordered_pilot_' + task + '.log')).open('ab') as log:
            p = subprocess.run([PYTHON, str(R / 'ops/run_task.py'), '--task', task,
                                '--attempt', '1', '--pilot'], stdout=log,
                               stderr=subprocess.STDOUT, timeout=min(
                                   960, max(1, C['model_deadline_epoch'] - time.time())))
            assert p.returncode == 0, 'Full-size CPU pilot failed; do not admit broad jobs'
    value = json.loads((folder / 'RESULT.json').read_text())
    start = json.loads((folder / 'START.json').read_text())
    assert value['status'] == 'complete'
    assert value['prediction_sha256'] == sha(folder / 'prediction.npz')
    assert value['start_sha256'] == sha(folder / 'START.json')
    assert start['context_rows'] == 50000 and start['query_rows'] == 20000
    assert not start['outer_labels_loaded'] and not start['correction_labels_loaded']
    assert not start['query_labels_used_for_early_stopping']
    assert not start['pilot_parameters_changed'] and start['device'] == 'cpu'
    assert value['audit']['execution_backend'] == 'CPU'
    assert value['audit']['inner_preprocessor']['fit_rows'] == 42500
    assert value['audit']['full_preprocessor']['fit_rows'] == 50000
    assert value['audit']['parameters']['boosting_type'] == (
        'Ordered' if '_ordered__' in task else 'Plain')
    assert not value.get('weights_saved', False) and not value.get('KV_cache_saved', False)
    import numpy as np
    with np.load(folder / 'prediction.npz') as saved:
        probability = saved['raw']
        assert len(probability) == 20000 and np.isfinite(probability).all()
    return dict(task=task, seconds=value['seconds'], result_sha256=sha(folder / 'RESULT.json'),
                prediction_sha256=value['prediction_sha256'], start_sha256=value['start_sha256'])


def start_remote():
    # Existing verified fitting packets and dependencies remain in place.
    for name in ['EXTRA_DESIGNS.jsonl', 'TASK_OWNERS.json', 'CPU_ORDERED_PLAN.json',
                 'CPU_ORDERED_GATE.json']:
        run(['rsync', '-az', '-e', TRANSPORT, str(R / name), HOST + ':' + str(R) + '/'])
    run(['rsync', '-az', '-e', TRANSPORT, '--include=GATE*.json', '--exclude=*',
         str(R) + '/', HOST + ':' + str(R) + '/'])
    code = """from pathlib import Path
import hashlib,json
r=Path(ROOT)
p=r/'CPU2003_QUEUE_FINAL.json'
if p.exists():
 d=r/'execution_flag_history';d.mkdir(exist_ok=True);h=hashlib.sha256(p.read_bytes()).hexdigest();q=d/('CPU2003_QUEUE_FINAL.'+h[:16]+'.json');q.write_bytes(p.read_bytes());p.unlink()
""".replace('ROOT', repr(str(R)))
    run(SSH + ['python3 -'], input=code)
    command = [PYTHON, '-u', str(R / 'ops/worker_queue.py'), '--resource', 'cpu',
               '--slots', '2', '--folds', '3,4', '--name', 'CPU2003_QUEUE']
    runtime = max(1, int(C['model_deadline_epoch'] - time.time() + 15))
    remote = ('systemd-run --user --unit=logika-sixhour-cpu-ordered-20261028 '
              '--property=MemoryMax=20G --property=CPUQuota=400% '
              '--property=RuntimeMaxSec=' + str(runtime) + ' --working-directory='
              + str(R) + ' ' + shlex.join(command))
    run(SSH + [remote])
    dump(dict(UTC=utc(), service='logika-sixhour-cpu-ordered-20261028.service',
              command=command, original_model_deadline=C['model_deadline_epoch']),
         R / 'CPU_ORDERED_REMOTE_DISPATCH.json')


def collect_remote():
    mirror = R / 'private/node2003_ordered'
    mirror.mkdir(parents=True, exist_ok=True)
    run(['rsync', '-az', '-e', TRANSPORT, '--include=runs/***', '--include=dispatch/***',
         '--include=CPU2003_QUEUE*.json', '--exclude=*', HOST + ':' + str(R) + '/',
         str(mirror) + '/'], timeout=180)
    run(['rsync', '-az', str(mirror) + '/', str(R) + '/'])
    return (mirror / 'CPU2003_QUEUE_FINAL.json').exists()


def main():
    assert sha(R / 'CLOCK.json') == PLAN['clock_sha256']
    assert load('CPU_ORDERED_PREPUBLICATION.json')['remote_commit']
    assert time.time() < C['admission_end_epoch']
    retire_flag('CPU_QUEUE_FINAL.json')
    results = [pilot(task) for task in PLAN['gate']['pilots']]
    total = sum(p['seconds'] for p in results)
    conservative = total * PLAN['gate']['remaining_node2001_fits_per_scheme_after_pilots'] / 2 * PLAN['gate']['runtime_safety_factor']
    passed = (all(p['seconds'] <= PLAN['gate']['max_pilot_seconds_each'] for p in results)
              and time.time() + conservative <= PLAN['gate']['latest_projected_finish_epoch'])
    gate = dict(status='PASS' if passed else 'FAIL_RESOURCE_FEASIBILITY', UTC=utc(),
                pilots=results, conservative_remaining_wall_seconds=conservative,
                projected_finish_epoch=time.time() + conservative,
                latest_projected_finish_epoch=PLAN['gate']['latest_projected_finish_epoch'],
                full_size_boundary_and_finite_checks=True, feature_and_scientific_source_unchanged=True,
                model_quality_conclusion='None from runtime feasibility; scores not used for admission.',
                plan_sha256=sha(R / 'CPU_ORDERED_PLAN.json'), clock_sha256=sha(R / 'CLOCK.json'))
    dump(gate, R / 'CPU_ORDERED_GATE.json')
    if not passed:
        dump(dict(stage='stopped_addendum_not_admitted', UTC=utc(),
                  reason='Predeclared runtime gate failed; completed pilots retained'), R / 'CPU_QUEUE_FINAL.json')
        dump(dict(stage='runtime_gate_not_admitted', UTC=utc()), R / 'CPU_ORDERED_FINAL.json')
        return
    retire_flag('CPU2003_QUEUE_FINAL.json')
    retire_flag('NODE2003_COLLECTION_COMPLETE.json')
    start_remote()
    command = [PYTHON, '-u', str(R / 'ops/worker_queue.py'), '--resource', 'cpu',
               '--slots', '2', '--folds', '0,1,2', '--name', 'CPU_QUEUE']
    with (R / 'logs/CPU_ORDERED_LOCAL_QUEUE.log').open('ab') as log:
        local = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    dump(dict(UTC=utc(), pid=local.pid, command=command), R / 'CPU_ORDERED_LOCAL_DISPATCH.json')
    collected = False
    while time.time() < C['deadline_epoch'] - 300:
        try:
            collected = collect_remote()
            dump(dict(stage='running', UTC=utc(), local_queue_pid=local.pid,
                      local_returncode=local.poll(), remote_collected=collected,
                      scientific_source_and_clock_unchanged=True), R / 'CPU_ORDERED_STATE.json')
            if collected and local.poll() is not None:
                break
        except Exception:
            # Collection errors never cancel a healthy fitting process.
            (R / 'logs/CPU_ORDERED_COLLECTION_FAILURE.txt').write_text(traceback.format_exc())
            dump(dict(stage='collection_retry', UTC=utc(), healthy_fits_untouched=True),
                 R / 'CPU_ORDERED_STATE.json')
        time.sleep(25)
    if collected:
        dump(dict(stage='collection_complete', UTC=utc(), addendum=True,
                  source='node2003'), R / 'NODE2003_COLLECTION_COMPLETE.json')
    assert local.poll() is not None and collected, 'Need terminal process/output reconciliation; no completion fabricated'
    dump(dict(stage='complete', UTC=utc(), local_returncode=local.returncode,
              original_clock_unchanged=True), R / 'CPU_ORDERED_FINAL.json')


if __name__ == '__main__':
    with (R / 'cpu_ordered_addendum.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception:
            (R / 'logs/CPU_ORDERED_FAILURE.txt').write_text(traceback.format_exc())
            dump(dict(stage='execution_failure_needs_reconciliation', UTC=utc(),
                      diagnostic='logs/CPU_ORDERED_FAILURE.txt',
                      no_scientific_gain_claim=True), R / 'CPU_ORDERED_STATE.json')
            raise
