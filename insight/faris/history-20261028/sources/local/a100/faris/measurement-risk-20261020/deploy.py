"""Run on PC2; PC3 access and deployment originate here."""
from pathlib import Path
import subprocess,json

ROOT=Path(__file__).resolve().parent
host='faris@gpu-labkc.ub.ac.id'
ssh=['ssh','-T','-p','2003',host]
subprocess.run(['rsync','-a','--exclude=__pycache__',str(ROOT)+'/',host+':faris/measurement-risk-20261020/',
                '-e','ssh -p 2003'],check=True)
script=r'''
import json,hashlib,subprocess,time,shutil
from pathlib import Path
root=Path.home()/'faris/measurement-risk-20261020'
assert not (root/'state.json').exists(),'Do not duplicate or reset an existing study'
assert shutil.disk_usage(root).free>3*1024**3
assert json.loads((root/'LEAKAGE_BOUNDARY.json').read_text())['status']=='passed'
assert subprocess.check_output(['loginctl','show-user','faris','-p','Linger','--value'],text=True).strip()=='yes'
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
names=['export.py','model.py','run.py','validate.py','plan.json','RESEARCH.md','input/manifest.json','LEAKAGE_BOUNDARY.json']
(root/'seal.json').write_text(json.dumps(dict(files={name:sha(root/name) for name in names}),indent=2)+'\n')
unit='logika-measurement-risk-20261020.service'
args=['systemd-run','--user','--unit='+unit,'--property=CPUQuota=1000%',
      '--property=MemoryHigh=8G','--property=MemoryMax=10G','--property=MemorySwapMax=0',
      '--property=TasksMax=256','--property=KillMode=control-group',
      '--property=RuntimeMaxSec='+str(max(1,int(1792585871.823748-time.time()))),
      '--property=StandardOutput=append:'+str(root/'controller.log'),
      '--property=StandardError=append:'+str(root/'controller.log'),
      '/mnt/home/faris/faris/exploration-12h-20261017/.venv/bin/python','-u',str(root/'run.py'),'run']
subprocess.run(args,check=True)
settings=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID',
    '-p','CPUQuotaPerSecUSec','-p','MemoryHigh','-p','MemoryMax','-p','MemorySwapMax','-p','ControlGroup'],text=True)
record=dict(status='launched',time=time.time(),settings=settings,deadline=1792585871.823748,
            root=str(root),unrelated_processes_touched=False,scripted_deadline_preserved=True)
(root/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
'''
result=subprocess.run(ssh+['python3','-'],input=script,text=True,capture_output=True,check=True)
print(result.stdout,flush=True)
(ROOT/'REMOTE_LAUNCH.json').write_text(result.stdout)
