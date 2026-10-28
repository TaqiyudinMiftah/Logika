"""Owned resource worker queue; scientifically sealed tasks, no score-based edits."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import check_seal,load,tasks,catalog,dump,utc
p=argparse.ArgumentParser();p.add_argument('--resource',choices=['cpu','gpu'],required=True);p.add_argument('--folds',default='0,1,2,3,4');p.add_argument('--slots',type=int,default=2);p.add_argument('--name',default='queue');p.add_argument('--owner',type=int);a=p.parse_args()
clock=check_seal();folds={int(x)for x in a.folds.split(',')};active={};failed=[]
(R/'logs').mkdir(exist_ok=True)
# Broad qualification always follows the predeclared family pilot on fold0.
def eligible():
 config=load('CONFIG.json');alltasks=tasks();designs=catalog();output=[]
 for key,t in alltasks.items():
  d=designs[t['design']]
  if d['resource']!=a.resource or t['fold']not in folds:continue
  if a.owner is not None and load('TASK_OWNERS.json')['designs'].get(t['design'])!=a.owner:continue
  attempt=int(load('APPROVED_RETRIES.json').get(key,1))if(R/'APPROVED_RETRIES.json').exists()else 1
  assert attempt in {1,2}
  if (R/'runs'/key/f'attempt{attempt}').exists()or(R/'dispatch'/key/f'attempt{attempt}').exists()or key in active:continue
  pilot=key in config['pilot_tasks']
  g=R/('GATE_FAMILY_'+d['family']+'.json')
  if not pilot and (not g.exists()or json.loads(g.read_text())['status']!='PASS'):continue
  # Queue CPU pilots first. For GPUs declared pilot first; complete designs in
  # catalog order across all5folds before spending on later architectures.
  order=load('EXECUTION_PRIORITY.json')['order']if(R/'EXECUTION_PRIORITY.json').exists()else list(designs)
  order=order+[name for name in designs if name not in order]
  output.append((0 if pilot else 1,order.index(t['design']),config['seeds'].index(t['seed']),t['fold'],key,pilot,attempt))
 return sorted(output)
try:
 while time.time()<clock['model_deadline_epoch']:
  for key,(process,handle)in list(active.items()):
   if process.poll()is not None:
    handle.close();del active[key]
    if process.returncode:failed.append({'task':key,'returncode':process.returncode,'UTC':utc()})
  data=R/'GATE.json'
  if time.time()<clock['admission_end_epoch'] and data.exists()and json.loads(data.read_text())['status']=='PASS':
   for *_,key,pilot,attempt in eligible()[:max(0,a.slots-len(active))]:
    handle=(R/'logs'/('dispatch_'+key+'.log')).open('ab');command=[sys.executable,str(R/'ops/run_task.py'),'--task',key,'--attempt',str(attempt)]
    if pilot:command.append('--pilot')
    proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);active[key]=(proc,handle)
  dump(dict(stage='running',name=a.name,UTC=utc(),resource=a.resource,folds=sorted(folds),active=[dict(task=k,pid=v[0].pid)for k,v in active.items()],completed=len(list((R/'runs').glob('*/attempt*/RESULT.json'))),failed=failed,admission_closed=time.time()>=clock['admission_end_epoch']),R/(a.name+'_STATE.json'))
  intended=[key for key,t in tasks().items()if catalog()[t['design']]['resource']==a.resource and t['fold']in folds and(a.owner is None or load('TASK_OWNERS.json')['designs'].get(t['design'])==a.owner)]
  if not active and (time.time()>=clock['admission_end_epoch']or all((R/'runs'/key).exists()for key in intended)):break
  time.sleep(10)
finally:
 for key,(proc,handle)in active.items():
  if proc.poll()is None:
   import signal
   os.killpg(proc.pid,signal.SIGTERM)
  handle.close()
 dump(dict(stage='stopped',UTC=utc(),failed=failed),R/(a.name+'_FINAL.json'))
