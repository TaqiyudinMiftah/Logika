"""Owned Colab collection/release under the human's earlier20:00WIB ceiling.

Execution control only. Sealed CLOCK, models, recipes and selection are untouched.
Freeze only the verified queue process, let existing fits finish, collect, release.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parents[1]
PLAN = json.loads((R / 'EARLY_STOP_PLAN.json').read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.early.tmp')
    q.write_text(json.dumps(value, indent=2) + '\n')
    q.replace(p)


def controller(account):
    previous = sys.argv
    sys.argv = ['colab_controller.py', '--account', str(account)]
    spec = importlib.util.spec_from_file_location('owned_controller_' + str(account), R / 'ops/colab_controller.py')
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = previous
    assert module.sha(R / 'CLOCK.json') == PLAN['original_clock_sha256']
    module.C = dict(module.C)
    module.C['deadline_epoch'] = min(module.C['deadline_epoch'], PLAN['deadline_epoch'] - 15)
    return module


def stop_controller(account):
    subprocess.run(['systemctl', '--user', 'stop', f'logika-cross-colab-a{account}-20261030.service'],
                   check=True, timeout=35)


def listing(c):
    response = c.run([*c.PREFIX, 'sessions'], timeout=45)
    dump(c.P / 'EARLY_SESSIONS_LATEST_PRIVATE.json', dict(UTC=stamp(), stdout=response.stdout))
    return response.stdout


def publish_mirror(c):
    """Replay the retained local packet against canonical immutable hashes."""
    mirror = c.P / 'mirror'
    manifest = {}
    checked = 0
    for result in (mirror / 'runs').glob('*/attempt1/RESULT.json'):
        value = json.loads(result.read_text())
        assert sha(result.parent / 'prediction.npz') == value['prediction_sha256']
        assert sha(result.parent / 'START.json') == value['start_sha256']
        assert value['weights_saved'] is False
        for name in ('START.json', 'RESULT.json', 'prediction.npz'):
            path = result.parent / name
            manifest[str(path.relative_to(mirror / 'runs'))] = sha(path)
        checked += 1
    assert checked > 0
    upload = c.CANON + f'/private/uploads/account{c.N}/runs/'
    c.run(['/usr/bin/rsync', '-az', '--ignore-existing', '-e', 'ssh -i ' + c.KEY + ' -o BatchMode=yes',
           str(mirror / 'runs') + '/', c.LAB + ':' + upload], timeout=180)
    c.run(['ssh', '-i', c.KEY, '-o', 'BatchMode=yes', c.LAB,
           'python3 ' + c.CANON + '/ops/atomic_collect.py --owner account' + str(c.N) +
           ' --source ' + upload], timeout=180)
    code = ('import hashlib,json;from pathlib import Path;r=Path(' + repr(c.CANON + '/runs') + ');'
            'm=json.loads(' + repr(json.dumps(manifest)) + ');'
            'assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m.items());'
            'print(len(m))')
    response = c.run(['ssh', '-i', c.KEY, '-o', 'BatchMode=yes', c.LAB,
                      'python3 -c ' + shlex.quote(code)], timeout=90)
    assert int(response.stdout.strip()) == len(manifest)
    receipt = dict(UTC=stamp(), checked_fit_records=checked, checked_terminal_files=len(manifest),
                   local_and_canonical_hashes_match=True, original_clock_sha256=PLAN['original_clock_sha256'])
    dump(c.P / 'EARLY_COLLECTION_AUDIT.json', receipt)
    return receipt


def upload(c, path, name):
    c.run(['scp', '-q', '-i', c.KEY, '-o', 'BatchMode=yes', str(path), c.LAB + ':' + c.CANON + '/' + name], timeout=90)


def release(c, receipt, reason, full_remote_collection):
    before = listing(c)
    if c.NAME in before:
        response = c.run([*c.PREFIX, 'stop', '--session', c.NAME], timeout=90)
        dump(c.P / 'EARLY_RELEASE_RESPONSE_PRIVATE.json', dict(UTC=stamp(), returncode=response.returncode,
                                                            stdout=response.stdout, stderr=response.stderr))
    after = listing(c)
    assert c.NAME not in after, 'Owned session release not yet confirmed by authenticated API'
    dump(c.P / 'RELEASED.json', dict(UTC=stamp(), session=c.NAME, owned_only=True, authenticated_API_absent=True))
    final = dict(UTC=stamp(), account=c.N, session=c.NAME, released=True, reason=reason,
                 canonical_collected=True, local_hash_verified=True,
                 checked_fit_records=receipt['checked_fit_records'], full_remote_collection=full_remote_collection,
                 unavailable_unfinished_fit_not_counted=True, original_clock_sha256=PLAN['original_clock_sha256'],
                 early_stop_plan_sha256=sha(R / 'EARLY_STOP_PLAN.json'))
    dump(c.P / 'FINAL_STATE.json', final)
    upload(c, c.P / 'FINAL_STATE.json', f'BACKEND_COLLECTION_account{c.N}_COMPLETE.json')
    upload(c, c.P / 'EARLY_COLLECTION_AUDIT.json', f'COLAB_A{c.N}_EARLY_COLLECTION_AUDIT.json')
    return final


def retire_provider(account):
    c = controller(account)
    if (c.P / 'RELEASED.json').exists():
        return
    observed = listing(c)
    owned_line = next((line for line in observed.splitlines() if '[' + c.NAME + ']' in line), None)
    assert owned_line is None or 'Hardware: CPU' in owned_line, 'GPU still granted; preserve running work'
    stop_controller(account)
    evidence = {}
    for name in ('known_hosts', 'COLLECTION_FAILURE_PRIVATE.log', 'TRANSPORT_FAILURE_PRIVATE.log'):
        p = c.P / name
        if p.exists():
            preserved = c.P / ('PRE_PROVIDER_REPLACEMENT_' + name)
            if not preserved.exists():
                preserved.write_bytes(p.read_bytes())
            evidence[name] = sha(preserved)
    for start in (c.P / 'mirror/runs').glob('*/attempt1/START.json'):
        folder = start.parent
        if any((folder / name).exists() for name in ('RESULT.json', 'FAILURE.json', 'INTERRUPTION.json')):
            continue
        dump(folder / 'INTERRUPTION.json', dict(status='interrupted', UTC=stamp(), task=folder.parent.name,
             reason='Authenticated API replaced owned GPU backend with CPU; original GPU unfinished fit unavailable',
             no_automatic_retry=True, scientific_conclusion=False, start_sha256=sha(start)))
    receipt = publish_mirror(c)
    record = dict(UTC=stamp(), account=account, status='provider_replaced_GPU_with_CPU',
                  retained_completed_fits=receipt['checked_fit_records'], evidence_sha256=evidence,
                  changed_host_keys_not_bypassed=True, no_reallocation_or_fit_retry=True)
    dump(c.P / 'PROVIDER_REPLACEMENT.json', record)
    upload(c, c.P / 'PROVIDER_REPLACEMENT.json', f'COLAB_A{account}_PROVIDER_REPLACEMENT.json')
    final = release(c, receipt, 'provider_replacement', full_remote_collection=False)
    print(json.dumps(dict(account=account, status='retained_hash_verified_and_released',
                          completed_fits=final['checked_fit_records'])), flush=True)


def remote_status(c, action):
    code = '''from pathlib import Path
import os,signal,json,time,datetime
r=Path(ROOT);action=ACTION;queue=r/'QUEUE_DISPATCH.json'
def valid(pid,needle):
 p=Path('/proc')/str(pid)/'cmdline'
 return p.exists() and needle.encode() in p.read_bytes()
qp=json.loads(queue.read_text()).get('pid') if queue.exists() else None
queue_valid=qp and valid(qp,str(r/'ops/owner_queue.py'))
if action=='freeze' and queue_valid:os.kill(qp,signal.SIGSTOP)
workers=[]
if (r/'DISPATCH.json').exists():
 for item in json.loads((r/'DISPATCH.json').read_text()).get('dispatches',[]):
  if valid(item['pid'],str(r/'source/worker.py')) and valid(item['pid'],item['task']):workers.append(item)
if action=='finish':
 assert not workers, 'Healthy owned fits still running'
 if queue_valid:os.kill(qp,signal.SIGKILL)
 for start in (r/'runs').glob('*/attempt1/START.json'):
  if not any((start.parent/n).exists() for n in ('RESULT.json','FAILURE.json','INTERRUPTION.json')):
   (start.parent/'INTERRUPTION.json').write_text(json.dumps(dict(status='interrupted',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),task=start.parent.parent.name,reason='Owned early-stop ceiling; no completed result',no_automatic_retry=True,scientific_conclusion=False)))
 d=json.loads((r/'BACKEND_STATE.json').read_text()) if (r/'BACKEND_STATE.json').exists() else {}
 d.update(status='terminal_human_early_stop',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_fits=len(list((r/'runs').glob('*/attempt1/RESULT.json'))),running={},early_stop=True,no_clock_extension=True)
 if not (r/'BACKEND_FINAL.json').exists():(r/'BACKEND_FINAL.json').write_text(json.dumps(d,indent=2))
if action=='ceiling':
 for item in workers:
  if valid(item['pid'],str(r/'source/worker.py')) and valid(item['pid'],item['task']):os.killpg(item['pid'],signal.SIGTERM)
print(json.dumps(dict(queue_verified=bool(queue_valid),active_owned_workers=len(workers),action=action)))
'''.replace('ROOT', repr(c.VROOT)).replace('ACTION', repr(action))
    return json.loads(c.ssh(code, timeout=45).stdout)


def early_run():
    c = controller(2)
    while time.time() < PLAN['admission_end_epoch']:
        time.sleep(min(15, PLAN['admission_end_epoch'] - time.time()))
    if (c.P / 'RELEASED.json').exists():
        return
    stop_controller(2)
    dump(R / 'EARLY_FINISH_STATE.json', dict(UTC=stamp(), stage='freezing_owned_queue'))
    try:
        record = remote_status(c, 'freeze')
        dump(c.P / 'EARLY_QUEUE_FREEZE.json', record)
        while True:
            status = remote_status(c, 'observe')
            if not status['active_owned_workers']:
                break
            if time.time() >= PLAN['model_deadline_epoch']:
                remote_status(c, 'ceiling')
            time.sleep(10)
        remote_status(c, 'finish')
        c.collect()
        receipt = publish_mirror(c)
        final = release(c, receipt, 'human_20WIB_early_stop', full_remote_collection=True)
        dump(R / 'EARLY_FINISH_STATE.json', dict(UTC=stamp(), stage='collected_and_released', final=final))
    except Exception:
        p = R / 'private/EARLY_FINISH_FAILURE_PRIVATE.log'
        p.write_text(traceback.format_exc())
        dump(R / 'EARLY_FINISH_STATE.json', dict(UTC=stamp(), stage='collection_or_release_review_required',
                                               private_failure_sha256=sha(p)))
        raise


def release_guard():
    """Independent billing guard if the graceful collection path loses transport."""
    boundary = PLAN['model_deadline_epoch'] + 300
    while time.time() < boundary:
        time.sleep(min(15, boundary - time.time()))
    for account in (2, 11):
        c = controller(account)
        if (c.P / 'RELEASED.json').exists():
            continue
        stop_controller(account)
        receipt = publish_mirror(c)
        release(c, receipt, 'human_early_deadline_billing_guard_partial_remote_coverage',
                full_remote_collection=False)
    subprocess.run(['systemctl', '--user', 'stop', 'logika-cross-resources-20261030.service'],
                   check=True, timeout=30)
    dump(R / 'EARLY_RESOURCE_GUARD_COMPLETE.json', dict(UTC=stamp(),
         owned_sessions_released=True, original_clock_unchanged=True))


if __name__ == '__main__':
    if time.time() >= PLAN['deadline_epoch'] - 30:
        raise SystemExit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--retire-provider11', action='store_true')
    parser.add_argument('--guard', action='store_true')
    args = parser.parse_args()
    if args.retire_provider11:
        retire_provider(11)
    elif args.guard:
        release_guard()
    else:
        early_run()
