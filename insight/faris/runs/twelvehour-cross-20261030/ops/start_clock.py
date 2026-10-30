"""Start the newly authorized budget once, including allocation and calibration."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import time

R = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R / 'CLOCK.json').exists(), 'Never reset an existing campaign clock'
publication = json.loads((R / 'PREPUBLICATION.json').read_text())
assert publication['config_sha256'] == sha(R / 'CONFIG.json')
assert publication['source_seal_sha256'] == sha(R / 'SOURCE_SEAL.json')
for name in ['private','logs','runs','evaluation_inputs']:(R/name).mkdir(exist_ok=True)
start = time.time()
clock = dict(started_epoch=start, start_epoch=start,
             started_UTC=datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat(),
             admission_end_epoch=start + 10.5 * 3600,
             model_deadline_epoch=start + 11.25 * 3600, deadline_epoch=start + 12 * 3600,
             hours=12, authorization='Human explicitly chose12hours for a NEW campaign',
             config_sha256=sha(R / 'CONFIG.json'), contract_sha256=sha(R / 'VALIDATION_CONTRACT.json'),
             source_seal_sha256=sha(R / 'SOURCE_SEAL.json'), previous_campaign_clocks_unchanged=True,
             includes_allocation_setup_calibration_training_and_assessment=True)
with (R / 'CLOCK.json').open('x') as stream:
    json.dump(clock, stream, indent=2)
    stream.write('\n')
os.chmod(R / 'CLOCK.json', 0o444)
def timestamp(value):
    return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).isoformat()
(R / 'DEADLINE.md').write_text('\n'.join([
    '# Immutable twelve-hour ceiling', '', 'StartedUTC: ' + timestamp(start),
    'AdmissionUTC: ' + timestamp(clock['admission_end_epoch']),
    'ModelUTC: ' + timestamp(clock['model_deadline_epoch']),
    'FinalUTC: ' + timestamp(clock['deadline_epoch']),
    'Never reset or extend this clock. Preserve downtime and failed attempts.', '']))
(R / 'TRAINING_BRIEF.md').write_text(
    'NEW human-authorized12-hour Logika research campaign. Bound all work to CLOCK.json. '
    'Use registered cross/kernel configurations and measured gates only. '
    'Keep parameters, source, seeds and compact private predictions; no fitted weights or KV dumps. '
    'No automatic refit, Kaggle upload or leaderboard probe. All70k cases are reused development.\n')
print(json.dumps(clock))

