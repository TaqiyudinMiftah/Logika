"""Forced PC1 campaign command. All file operations confined to this new root."""
import json,sys,tarfile,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def inside(name):
 p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and p!=ROOT;return p
r=json.loads(sys.stdin.buffer.readline())
if r['action']=='put':
 dest=inside(r['destination']);dest.mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=sys.stdin.buffer,mode='r|*') as t:
  for m in t:
   p=(dest/m.name).resolve();assert m.isfile() and p.is_relative_to(dest) and not m.issym() and not m.islnk();p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('wb') as f:
    import shutil;shutil.copyfileobj(t.extractfile(m),f)
 print(json.dumps({'stored':True}))
elif r['action']=='activate_features':
 import shutil
 incoming=inside('incoming')
 manifest=json.loads((incoming/'extras/manifest.json').read_text())
 import hashlib
 for name,checksum in manifest.get('files',{}).items():assert hashlib.sha256((incoming/'extras'/name).read_bytes()).hexdigest()==checksum
 (ROOT/'extras/manifest.json').unlink(missing_ok=True)
 for name in ['extras','evidence']:shutil.copytree(incoming/name,ROOT/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('manifest.json') if name=='extras' else None)
 # Publish readiness after every feature file is present.
 shutil.copy2(incoming/'extras/manifest.json',ROOT/'extras/manifest.json')
 print('features_activated')
elif r['action']=='get':
 paths=r['paths']
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
  for name in paths:
   p=inside(name)
   for f in ([p] if p.is_file() else p.rglob('*')):
    if f.is_file():t.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
elif r['action']=='prune':
 from archive_ops import prune
 print(json.dumps(prune(r)))
elif r['action']=='pause_admission':
 inside('campaign/pause_admission.json').write_text(json.dumps(r));print('admission_paused')
elif r['action']=='tasks':
 p=inside('stage_tasks.json');old=json.loads(p.read_text()) if p.exists() else [];known={x['id'] for x in old};old += [x for x in r['tasks'] if x['id'] not in known];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(old));tmp.replace(p);print(json.dumps({'tasks':len(old)}))
elif r['action']=='clock':
 p=inside('clock.json');assert not p.exists() or json.loads(p.read_text())==r['clock'];p.write_text(json.dumps(r['clock']));print('ok')
elif r['action']=='start':
 host='pc1';cmd=['systemd-run','--user','--slice=explore12h.slice','--unit=explore12h-pc1','--property=Restart=on-failure','--property=RestartSec=15','--property=CPUQuota=1600%','--property=MemoryMax='+str(json.loads((ROOT/'deployment.json').read_text())['hosts']['pc1']['ram_mb'])+'M','--property=MemorySwapMax=0','--property=WorkingDirectory='+str(ROOT),'--property=StandardOutput=append:'+str(ROOT/'logs/controller.log'),'--property=StandardError=append:'+str(ROOT/'logs/controller.log'),str(ROOT/'.venv/bin/python'),str(ROOT/'bootstrap.py'),'pc1'];subprocess.run(cmd,check=True);print('started')
elif r['action']=='close':inside('campaign/close.json').write_text('{}');print('closed')
elif r['action']=='stop':inside('campaign/stop.json').write_text('{}');print('stop_requested')
elif r['action']=='control':
 action=r['command'];assert action in ['prepare','validate','resume','verify','replay']
 cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'host_prepare.py'),'pc1'] if action=='prepare' else [str(ROOT/'.venv/bin/python'),str(ROOT/'campaign/runtime/campaign.py'),action,str(ROOT/'campaign')]
 if action=='resume':
  from service import start
  print(start('pc1'));raise SystemExit(0)
 if action in ['verify','replay']:
  job=r['job'];assert job.replace('_','').isalnum();cmd+=[job]
 if action=='replay':cmd+=['--output',str(inside('replays/'+r['output']))]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=120);print(p.stdout);print(p.stderr if p.returncode else '');sys.exit(p.returncode)
elif r['action']=='status':
 p=ROOT/'campaign/state.json';s=json.loads(p.read_text()) if p.exists() else {'status':'preparing','jobs':{}}
 print(json.dumps(s))
else:raise ValueError('Unsupported campaign action')
