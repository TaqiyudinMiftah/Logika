"""Prospective routing by granted resources, never by model scores."""
from pathlib import Path
import hashlib
import json
import random
import time

R = Path(__file__).resolve().parents[1]
config = json.loads((R / 'CONFIG.json').read_text())
assert not (R / 'EXECUTION_PLAN.json').exists(), 'Do not rewrite a live ownership schedule'
owners = {name: dict(resource='cpu', slots=slots, designs=[])
          for name, slots in [('cpu2001', 4), ('cpu2003', 4), ('cpu119', 6), ('cpu42', 6)]}
owners['gpu2003'] = dict(resource='gpu', slots=2, gpu_indices=[0, 1], designs=[])
for account in range(1, 12):
    if (R / f'private/colab/account{account}/ALLOCATED_READY.json').exists():
        owners[str(account)] = dict(resource='gpu', slots=1, gpu_indices=[0], designs=[],
                                    requested_gpu='A100' if account == 2 else 'T4',
                                    explicitly_owned_session=f'logika-cross-a{account}-20261030')
rng = random.Random(20261030)
cpu = [key for key, value in config['designs'].items() if value['resource'] == 'cpu']
gpu = [key for key, value in config['designs'].items() if value['resource'] == 'gpu']
rng.shuffle(cpu)
rng.shuffle(gpu)
baseline = config['baseline_pilot_task'].split('__s', 1)[0]
cpu.remove(baseline)
owners['cpu2001']['designs'].append(baseline)
for key in cpu:
    name = min([x for x in owners if owners[x]['resource'] == 'cpu'],
               key=lambda x: (len(owners[x]['designs']) / owners[x]['slots'], x))
    owners[name]['designs'].append(key)
for key in gpu:
    eligible = [x for x in owners if owners[x]['resource'] == 'gpu']
    if config['designs'][key]['view'] == 'all':
        name = '2' if '2' in eligible else 'gpu2003'
    else:
        def weight(owner):
            return 2. if owner in {'2', 'gpu2003'} else .5
        name = min(eligible, key=lambda x: (len(owners[x]['designs']) / weight(x), x))
    owners[name]['designs'].append(key)
assigned = [key for owner in owners.values() for key in owner['designs']]
assert len(assigned) == len(set(assigned)) == len(config['designs'])
assert set(assigned) == set(config['designs'])
plan = dict(owners=owners, resource_assignment_uses_scores=False,
            whole_designs_keep_one_backend=True, CPU_threads_per_fit=2,
            full_bank_priority='OwnedA100 if granted; otherwise idleRTX3060 on2003',
            no_2001_GPU_use=True, no_2002_use=True, no_quota_retry_or_profile_rotation=True)
(R / 'EXECUTION_PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
seal = dict(epoch=time.time(), execution_plan_sha256=sha(R / 'EXECUTION_PLAN.json'),
            clock_sha256=sha(R / 'CLOCK.json'), config_sha256=sha(R / 'CONFIG.json'),
            source_seal_sha256=sha(R / 'SOURCE_SEAL.json'),
            before_any_model_fit=True, no_validation_results_used=True)
(R / 'DISPATCH_SEAL.json').write_text(json.dumps(seal, indent=2) + '\n')
print(json.dumps({name: dict(slots=item['slots'], designs=len(item['designs']))
                  for name, item in owners.items()}))
