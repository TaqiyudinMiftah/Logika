"""Collect immutable completed PC3 models; only after all30 fit the five heads."""
from pathlib import Path
import os,time,json,subprocess,sys,shutil
from context_model import sha
R=Path(__file__).resolve().parents[1];p=json.loads((R/'training-plan.json').read_text());D=p['original_deadline'];remote='/mnt/home/faris/faris/nested-correction-20261021';target='faris@gpu-labkc.ub.ac.id';needed=p['host_jobs']['pc3']
def valid(d):
 try:
  c=json.loads((d/'complete.json').read_text());return c['status']=='complete'and all(sha(d/n)==h for n,h in c['files'].items())
 except(FileNotFoundError,ValueError):return False
(R/'runs').mkdir(exist_ok=True);(R/'incoming').mkdir(exist_ok=True)
while True:
 assert time.time()<D and not(R/'STOP').exists()
 for j in needed:
  dest=R/f'runs/job{j}'
  if dest.exists():assert valid(dest),'Retained destination integrity failure';continue
  cmd='test -f '+remote+f'/runs/job{j}/complete.json';ready=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p2003',target,cmd],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  if ready.returncode:continue
  assert shutil.disk_usage(R).free>3*1024**3
  incoming=R/f'incoming/job{j}'
  if not valid(incoming):
   incoming.mkdir(exist_ok=True);code=subprocess.call(['scp','-P2003','-r',target+':'+remote+f'/runs/job{j}/.',str(incoming)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   if code or not valid(incoming):continue
  incoming.replace(dest)
 done=[j for j in range(30)if(R/f'runs/job{j}/complete.json').exists()];(R/'COLLECTION.json').write_text(json.dumps(dict(status='collecting',completed=done,missing=[j for j in range(30)if j not in done],original_deadline=D),indent=2))
 if len(done)==30:
  assert all(valid(R/f'runs/job{j}')for j in range(30));break
 # A terminal failure is reported rather than hidden behind an endless wait.
 local=json.loads((R/'TRAINING_STATUS.json').read_text())if(R/'TRAINING_STATUS.json').exists()else{}
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p2003',target,'cat '+remote+'/TRAINING_STATUS.json'],capture_output=True,text=True)
 try:other=json.loads(r.stdout)if r.returncode==0 else{}
 except ValueError:other={}
 if local.get('status')in['complete','partial']and other.get('status')in['complete','partial']and 'partial'in[local.get('status'),other.get('status')]:raise RuntimeError('All controllers terminal with incomplete30job coverage; models retained, no score')
 time.sleep(10)
with(R/'assembly.log').open('ab',buffering=0)as log:subprocess.run([sys.executable,str(R/'source/assemble.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
with(R/'evaluation.log').open('ab',buffering=0)as log:subprocess.run([sys.executable,str(R/'source/report.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
(R/'COLLECTION.json').write_text(json.dumps(dict(status='complete',jobs=30,heads=5,original_deadline=D),indent=2))
