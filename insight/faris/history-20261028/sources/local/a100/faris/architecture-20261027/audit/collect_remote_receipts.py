"""Capture read-only remote evidence identities and verify the local mirror."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/mnt/home/faris/logika-architecture-20261027'

code = r'''
from pathlib import Path
import datetime,hashlib,json,subprocess
r=Path('/mnt/home/faris/logika-architecture-20261027')
old=Path('/mnt/home/faris/logika-frozen-shadow-20261026/frozen')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(r)):h(p)for pattern in ['runs/**/*','pilots/**/*','source/*.py','evaluation/*']for p in r.glob(pattern)if p.is_file()and '__pycache__'not in p.parts}
for name in ['CLOCK.json','CONFIGS.json','VALIDATION_CONTRACT.json','FEATURE_FREEZE.json','GATE.json','GATE_PREP.json','MODEL_RUN_AUDIT.json','LAUNCH_AUDIT.json']:
 if (r/name).exists():files[name]=h(r/name)
seal=json.loads((r/'FEATURE_FREEZE.json').read_text())
for f in range(5):
 p=old/f'fold{f}';m=json.loads((p/'MANIFEST.json').read_text())
 assert h(p/'MANIFEST.json')==seal['folds'][f]['manifest_sha256']
 for name,key in [('features.parquet','features_sha256'),('roles.npz','roles_sha256'),('baseline.npz','baseline_sha256')]:assert h(p/name)==m[key]
units=subprocess.check_output(['systemctl','--user','list-units','--all','--no-legend','--plain','logika-architecture-*'],text=True)
assert not units.strip(),units
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'all5_original_feature_role_baseline_hashes_verified':True,'active_architecture_units':units.strip()}))
'''

node1 = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'faris@100.68.29.126']
one = subprocess.run(node1 + ['python3 -c ' + shlex.quote(code)], capture_output=True, text=True, check=True, timeout=90)
hop = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2003','faris@gpu-labkc.ub.ac.id', 'python3 -c ' + shlex.quote(code)]
three = subprocess.run(node1 + [shlex.join(hop)], capture_output=True, text=True, check=True, timeout=90)
receipts = {'node2001':json.loads(one.stdout), 'node2003':json.loads(three.stdout)}
for host, receipt in receipts.items():
    for filename, expected in receipt['files'].items():
        local = ROOT / filename
        assert local.exists(), (host, filename, 'missing local copy')
        assert hashlib.sha256(local.read_bytes()).hexdigest() == expected, (host, filename, 'hash mismatch')
    count = len([p for p in receipt['files'] if p.endswith('/RESULT.json') and p.startswith('runs/')])
    receipt['local_mirror_verified'] = True
    receipt['main_fit_results_on_host'] = count
assert receipts['node2001']['main_fit_results_on_host'] == 160
assert receipts['node2003']['main_fit_results_on_host'] == 72
out = ROOT / 'REMOTE_RECEIPTS.json'
out.write_text(json.dumps(receipts, indent=2)+'\n')
print(json.dumps({k:{field:v[field]for field in ['UTC','all5_original_feature_role_baseline_hashes_verified','active_architecture_units','local_mirror_verified','main_fit_results_on_host']}for k,v in receipts.items()}, indent=2))
