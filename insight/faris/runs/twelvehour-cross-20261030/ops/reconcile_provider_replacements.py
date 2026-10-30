"""Retire only authenticated owned GPU-to-CPU replacements; never bypass host keys."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time

R = Path('/home/faris/logika-twelvehour-cross-20261030')
CANON = '/mnt/home/faris/logika-twelvehour-cross-20261030'
os.environ['PATH'] = '/home/faris/.local/bin:' + os.environ['PATH']
CLI = '/home/faris/.local/bin/colab'
KEY = '/home/faris/.ssh/id_ed25519_logika_lab'
ACCOUNTS = [1, 3, 4, 6, 8, 9, 10]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def one(n):
    p = R / f'private/colab/account{n}'
    name = f'logika-cross-a{n}-20261030'
    prefix = [CLI, '--account', str(n), '--config', str(p / 'sessions.json')]
    assert not (p / 'TRAINING_STARTED.json').exists(), 'Reconcile scientific queue before retiring'
    assert not list((p / 'mirror/runs').glob('*/attempt1/RESULT.json'))
    before = subprocess.run([*prefix, 'sessions'], capture_output=True, text=True, timeout=90)
    dump(p / 'PROVIDER_RECONCILIATION_BEFORE.json', dict(epoch=time.time(), returncode=before.returncode,
         stdout=before.stdout, stderr=before.stderr))
    before.check_returncode()
    owned = [line for line in before.stdout.splitlines() if '[' + name + ']' in line]
    assert len(owned) == 1 and 'Hardware: CPU' in owned[0], 'Never release an unknown or healthy GPU'
    subprocess.run(['systemctl', '--user', 'stop', f'logika-cross-colab-a{n}-20261030'],
                   check=True, timeout=40)
    response = subprocess.run([*prefix, 'stop', '--session', name], capture_output=True,
                              text=True, timeout=120)
    dump(p / 'RELEASE_RESPONSE.json', dict(epoch=time.time(), returncode=response.returncode,
         stdout=response.stdout, stderr=response.stderr))
    after = subprocess.run([*prefix, 'sessions'], capture_output=True, text=True, timeout=90)
    dump(p / 'POST_RELEASE_SESSIONS.json', dict(epoch=time.time(), returncode=after.returncode,
         stdout=after.stdout, stderr=after.stderr))
    after.check_returncode()
    assert response.returncode == 0 and '[' + name + ']' not in after.stdout
    record = dict(epoch=time.time(), account=n, session=name,
                  status='provider_replaced_GPU_with_CPU_during_setup', released=True,
                  authenticated_hardware='CPU', previous_allocated_hardware='T4',
                  actual_scientific_queue_dispatched=False, collected_completed_fits=0,
                  predictive_conclusion=False, no_host_key_bypass=True, no_reallocation=True,
                  old_known_hosts_preserved=(p / 'known_hosts').exists(),
                  preflight_sha256=hashlib.sha256((p / 'PROVIDER_RECONCILIATION_BEFORE.json').read_bytes()).hexdigest(),
                  release_evidence_sha256=hashlib.sha256((p / 'POST_RELEASE_SESSIONS.json').read_bytes()).hexdigest(),
                  original_deadline_epoch=json.load(open(R / 'CLOCK.json'))['deadline_epoch'])
    dump(p / 'PROVIDER_REPLACEMENT.json', record)
    dump(p / 'RELEASED.json', record)
    dump(p / 'FINAL_STATE.json', record)
    for filename, target in [('PROVIDER_REPLACEMENT.json', f'COLAB_A{n}_PROVIDER_REPLACEMENT.json'),
                             ('FINAL_STATE.json', f'BACKEND_COLLECTION_account{n}_COMPLETE.json')]:
        subprocess.run(['scp', '-q', '-i', KEY, '-o', 'BatchMode=yes', str(p / filename),
                        'faris@100.68.29.126:' + CANON + '/' + target], check=True, timeout=45)
    print(json.dumps(record), flush=True)
    return record


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    values = list(executor.map(one, ACCOUNTS))
dump(R / 'PROVIDER_REPLACEMENT_RECONCILIATION.json', dict(epoch=time.time(), accounts=values,
     no_scientific_source_or_clock_change=True, healthy_backends_untouched=True))
