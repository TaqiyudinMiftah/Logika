"""Finite PC2 artifact collector. It never schedules or stops training."""
import subprocess,json,time,hashlib,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HOST='faris@gpu-labkc.ub.ac.id'
SSH=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','2003',HOST]
DEADLINE=1792585871.823748
while time.time()<DEADLINE+120:
    try:
        probe=subprocess.run(SSH+['python3','-'],input="""
import json,subprocess
from pathlib import Path
p=Path.home()/'faris/measurement-risk-20261020'
s=json.loads((p/'state.json').read_text())
s['unit_status']=subprocess.check_output(['systemctl','--user','show','logika-measurement-risk-20261020.service','-p','ActiveState','--value'],text=True).strip()
print(json.dumps(s))
""",capture_output=True,text=True,timeout=20,check=True)
        state=json.loads(probe.stdout)
        if state['unit_status'] not in ['active','activating','deactivating']:break
    except (subprocess.SubprocessError,ValueError):pass
    time.sleep(20)
else:raise RuntimeError('Collector deadline exceeded; PC3 artifacts retained')
manifest_script="""
import hashlib,json
from pathlib import Path
p=Path.home()/'faris/measurement-risk-20261020'
d={}
for f in sorted(p.rglob('*')):
 if f.is_file() and '__pycache__' not in f.parts:
  with f.open('rb') as stream:d[str(f.relative_to(p))]=hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps(d))
"""
try:
    remote=subprocess.check_output(SSH+['python3','-'],input=manifest_script,text=True,timeout=120)
    manifest=json.loads(remote);dest=ROOT/'PC3-results';dest.mkdir(exist_ok=True)
    subprocess.run(['rsync','-a','--exclude=__pycache__','-e','ssh -p 2003',
        HOST+':faris/measurement-risk-20261020/',str(dest)+'/'],check=True,timeout=180)
    for name,digest in manifest.items():
        with (dest/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
    receipt=dict(status='verified',time=time.time(),files_verified=len(manifest),remote_state=state,
                 manifest=manifest,models_deleted=False)
    (ROOT/'COLLECTION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='verified',files=len(manifest),training_status=state['status'])),flush=True)
except BaseException:
    (ROOT/'COLLECTION_FAILURE.txt').write_text(traceback.format_exc());raise
