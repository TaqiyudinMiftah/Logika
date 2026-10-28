"""Complete measured GPU family qualifications; no scoring or model changes."""
from pathlib import Path
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import load,tasks,catalog,check_seal,sha,dump,utc
C=check_seal();tried={}
while time.time()<C['model_deadline_epoch']:
 for key in load('CONFIG.json')['pilot_tasks']:
  d=catalog()[tasks()[key]['design']]
  if d['resource']!='gpu':continue
  family=d['family'];g=R/('GATE_FAMILY_'+family+'.json')
  if g.exists()and json.loads(g.read_text())['status']=='PASS':continue
  completed=list((R/'runs'/key).glob('attempt*/RESULT.json'))
  if not completed or not(R/'GPU_BACKENDS.json').exists():continue
  p=completed[0];start=json.loads((p.parent/'START.json').read_text());bindings=load('GPU_BACKENDS.json')
  if start['host']not in bindings:continue
  identity=sha(p)
  if tried.get(family)==identity:continue
  with(R/'logs'/('qualified_'+family+'.log')).open('ab')as log:
   code=subprocess.run([str(R/'env311/bin/python'),'ops/qualify_gpu.py','--family',family],stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
  tried[family]=identity
  dump(dict(UTC=utc(),family=family,returncode=code,result_sha256=identity),R/('QUALIFIER_'+family+'.json'))
 owned=load('TASK_OWNERS.json')['owned_sessions']if(R/'TASK_OWNERS.json').exists()else[]
 flags=[R/f'GPU_COLLECTION_VM{n}_COMPLETE.json'for n in owned]
 if flags and all(p.exists()and json.loads(p.read_text()).get('released')for p in flags):
  dump(dict(UTC=utc(),owned=owned,all_released=True,evidence={p.name:sha(p)for p in flags}),R/'GPU_COLLECTION_COMPLETE.json')
 time.sleep(15)
