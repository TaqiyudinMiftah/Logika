"""Reconcile only owned work and preserve the superseded lab Kumo cohort."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import check_seal, dump, load, sha, utc
from coordinator import probe, collect

clock = check_seal()
assert load('COLAB_REPLAY_GATE.json')['status'] == 'PASS'
assert time.time() < clock['admission_end_epoch']
receipt = R / 'COLAB_CUTOVER.json'
if receipt.exists():
    assert load('COLAB_CUTOVER.json')['status'] == 'complete'
    print(receipt.read_text())
    raise SystemExit(0)

subprocess.run(['systemctl', '--user', 'stop', 'logika-feature-subspace-moe-20261028.service'], check=True)
tasks = load('CONFIG.json')['tasks']
records = []
for folder in (R / 'dispatch').glob('*'):
    if tasks[folder.name]['family'] == 'kumo':
        for file in folder.glob('attempt*.json'):
            records.append(json.loads(file.read_text()))
# Workers are distinct systemd units. Never stop a healthy fit; reconcile it.
for job in records:
    while True:
        status = probe(job)
        if not status['active']:
            if status['result'] or status['failure']:
                collect(job)
            break
        assert time.time() < clock['deadline_epoch'] - 7200
        time.sleep(15)

preserved = {}
for task, config in tasks.items():
    if config['family'] != 'kumo':
        continue
    for category, retired in [('runs', 'retired_lab_runs'), ('dispatch', 'retired_lab_dispatch')]:
        source, destination = R / category / task, R / retired / task
        if source.exists():
            assert not destination.exists(), 'Never overwrite retired evidence'
            for file in source.rglob('*'):
                if file.is_file():
                    preserved[str(file.relative_to(R))] = sha(file)
            destination.parent.mkdir(exist_ok=True)
            source.rename(destination)
dump({'status': 'complete', 'UTC': utc(), 'original_clock_sha256': sha(R / 'CLOCK.json'),
      'retired_artifact_hashes': preserved, 'retirement_reason': 'Consistent A100 Kumo backend cohort; never combine old lab outputs into replacement fits.',
      'CPU_expert_fits_retained': 180, 'unrelated_workloads_stopped': False,
      'scientific_source_config_contract_changed': False}, receipt)
dump({'stage': 'modelling', 'backend': 'Colab account2 A100', 'completed': 180,
      'expected': 260, 'pending': 80, 'active': [], 'failed': 0, 'failures': [],
      'UTC': utc(), 'clock': clock, 'cutover_evidence': 'COLAB_CUTOVER.json'}, R / 'LIVE_STATE.json')
print(receipt.read_text())
