"""Persistent canonical watchdog for the explicitly earlier human deadline."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import time

R = Path(__file__).resolve().parents[1]
P = json.loads((R / 'EARLY_STOP_PLAN.json').read_text())


def dump(name, value):
    target = R / name
    temporary = target.with_name(target.name + '.early.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(target)


while time.time() < P['deadline_epoch'] - 20:
    assert hashlib.sha256((R / 'CLOCK.json').read_bytes()).hexdigest() == P['original_clock_sha256']
    if time.time() >= P['model_deadline_epoch'] + 180 and not (R / 'COMPUTATION_FINAL.json').exists():
        owners = json.loads((R / 'EXECUTION_PLAN.json').read_text())['owners']
        absent = [owner for owner in owners if not (R / ('COLLECTED_' + owner + '.json') if owner.startswith(('cpu', 'gpu'))
                   else R / ('BACKEND_COLLECTION_account' + owner + '_COMPLETE.json')).exists()]
        dump('COMPUTATION_FINAL.json' if not absent else 'EARLY_FINALIZATION_WAIT.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             all_owner_collections_verified=not absent, unavailable_owners=absent,
             human_early_stop=True, early_stop_plan_sha256=hashlib.sha256((R/'EARLY_STOP_PLAN.json').read_bytes()).hexdigest(),
             no_unknown_attempts_retried=True, incomplete_owners_not_claimed_complete=True))
    time.sleep(10)

units = ['logika-cross-' + name + '-20261030.service' for name in
         ('coordinator', 'compose', 'evaluate', 'checkpoint42', 'cpu2001')]
subprocess.run(['systemctl', '--user', 'stop', *units], check=True, timeout=18)
dump('EARLY_CEILING_REACHED.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
     human_deadline_epoch=P['deadline_epoch'], completed=(R/'COMPLETION.json').exists(),
     only_owned_campaign_units_stopped=units, original_clock_unchanged=True))
