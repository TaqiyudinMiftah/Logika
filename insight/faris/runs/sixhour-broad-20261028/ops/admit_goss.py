"""Wait for ordinary sanitized publication, then admit the predeclared pilot gate."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess
import time

R = Path(__file__).resolve().parents[1]
REPO = R.parents[2] / '.history-publish-20261028'
REMOTE = '/mnt/home/faris/logika-sixhour-broad-20261028'
NAMESPACE = 'insight/faris/runs/sixhour-broad-20261028/'
clock = json.loads((R / 'CLOCK.json').read_text())

def run(args, **kw):
    return subprocess.run(args, text=True, capture_output=True, check=True, timeout=45, **kw)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert REPO.is_dir(), REPO
assert not (R / 'CPU_GOSS_PREPUBLICATION.json').exists()
while time.time() < clock['admission_end_epoch'] - 100:
    receipt = json.loads((R / 'private/publication/LATEST_RECEIPT.json').read_text())
    commit = receipt['remote_commit']
    try:
        plan = json.loads(run(['git', 'show', commit + ':' + NAMESPACE + 'CPU_GOSS_PLAN.json'], cwd=REPO).stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        time.sleep(5)
        continue
    assert plan == json.loads((R / 'CPU_GOSS_PLAN.json').read_text())
    published = [json.loads(line) for line in run(['git', 'show', commit + ':' + NAMESPACE + 'EXTRA_DESIGNS.jsonl'], cwd=REPO).stdout.splitlines() if line.strip()]
    local = [json.loads(line) for line in (R / 'EXTRA_DESIGNS.jsonl').read_text().splitlines() if line.strip()]
    assert published == local
    hashes = {}
    for name in ['CLOCK.json', 'source/worker.py', 'source/models_classical.py']:
        value = subprocess.run(['git', 'show', commit + ':' + NAMESPACE + name], cwd=REPO, capture_output=True, check=True).stdout
        hashes[name] = hashlib.sha256(value).hexdigest()
        assert hashes[name] == sha(R / name)
    actual = run(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=REPO).stdout.split()[0]
    assert actual == commit, 'Remote changed; reconcile before fitting'
    evidence = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    remote_commit=commit, remote_HEAD_verified=actual,
                    preregistered_plan_and_ledger_semantically_identical=True,
                    scientific_source_and_clock_hashes=hashes,
                    operational_helper_sha256=sha(R / 'ops/cpu_goss_addendum.py'),
                    no_fitting_before_this_verified_publication=True)
    path = R / 'CPU_GOSS_PREPUBLICATION.json'
    path.write_text(json.dumps(evidence, indent=2) + '\n')
    run(['rsync', '-az', str(path), 'faris@100.68.29.126:' + REMOTE + '/'])
    runtime = max(1, int(clock['model_deadline_epoch'] - time.time() + 20))
    command = ['systemd-run', '--user', '--unit=logika-sixhour-cpu-goss-addendum-20261028',
               '--property=MemoryMax=20G', '--property=CPUQuota=400%',
               '--property=KillMode=process', '--property=RuntimeMaxSec=' + str(runtime),
               '--working-directory=' + REMOTE, REMOTE + '/env311/bin/python',
               '-u', REMOTE + '/ops/cpu_goss_addendum.py']
    result = run(['ssh', '-oBatchMode=yes', 'faris@100.68.29.126', shlex.join(command)])
    print(json.dumps(dict(stage='runtime_gate_launched', publication=commit, UTC=evidence['UTC'])), flush=True)
    break
else:
    print(json.dumps(dict(stage='not_launched', reason='Publication not verified early enough; no clock extension or fitting.')), flush=True)
