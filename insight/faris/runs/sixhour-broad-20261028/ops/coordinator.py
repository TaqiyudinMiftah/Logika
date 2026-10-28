"""Canonical lab coordinator. Execute only predeclared tasks, preserve the clock."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import check_seal,load,dump,utc,sha,tasks,catalog
clock=check_seal();cpu=None;last_eval=0;data_tried=(R/'GATE.json').exists()and load('GATE.json')['status']=='PASS';family_tried={};last_count=-1
(R/'logs').mkdir(exist_ok=True)
def command(args,name,timeout=900):
 with (R/'logs'/name).open('ab')as log:return subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
try:
 while time.time()<clock['model_deadline_epoch']:
  if (R/'PREPARATION_AUDIT.json').exists()and not data_tried:
   code=command(['source/gate.py','--data'],'gate_data.log');data_tried=True
   if code:raise RuntimeError('Measured data admission failed; private gate diagnostics retained')
  gate=R/'GATE.json';admitted=gate.exists()and load('GATE.json')['status']=='PASS'
  if admitted:
   if cpu is None:
    log=(R/'logs/CPU_QUEUE.log').open('ab');cpu=subprocess.Popen([str(R/'env311/bin/python'),'ops/queue.py','--resource','cpu','--slots','2','--folds','0,1,2','--name','CPU_QUEUE'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   for task in load('CONFIG.json')['pilot_tasks']:
    family=catalog()[tasks()[task]['design']]['family'];paths=sorted((R/'runs'/task).glob('attempt*/RESULT.json'))
    if paths and family not in family_tried:
     family_tried[family]=command(['source/gate.py','--family',family],'gate_'+family+'.log')
   count=len(list((R/'runs').glob('*/attempt*/RESULT.json')))
   if count!=last_count and time.time()-last_eval>180:
    command(['source/evaluate.py'],'evaluation.log',timeout=900);last_count=count;last_eval=time.time()
  else:count=0
  dump(dict(stage='modelling'if admitted else'preparing_features_and_data_gate',UTC=utc(),expected=len(tasks()),completed=count,failed_attempts=len(list((R/'runs').glob('*/attempt*/FAILURE.json'))),data_gate=load('GATE.json').get('status')if gate.exists()else'pending',family_gates={f:load('GATE_FAMILY_'+f+'.json')['status']for f in family_tried},CPU_queue_pid=cpu.pid if cpu else None,clock_sha256=sha(R/'CLOCK.json'),independent_audit_available=False),R/'LIVE_STATE.json')
  time.sleep(15)
 # Stop admitting at immutable deadline; run_task owns per-fit bounded timeout.
 if cpu:cpu.wait(timeout=60)
 gpu=R/'GPU_COLLECTION_COMPLETE.json'
 # Collection reserve lies strictly inside the immutable total ceiling.
 while (not gpu.exists() or not(R/'NODE2003_COLLECTION_COMPLETE.json').exists())and time.time()<clock['deadline_epoch']-900:time.sleep(10)
 if (R/'GATE.json').exists()and load('GATE.json')['status']=='PASS':
  code=command(['source/evaluate.py','--final'],'final_evaluation.log',timeout=max(10,int(clock['deadline_epoch']-time.time()-30)))
  if code:raise RuntimeError('Final evaluation failed; no completion fabricated')
 completed=len(list((R/'runs').glob('*/attempt*/RESULT.json')))
 dump(dict(stage='complete_with_partial_designs'if completed<len(tasks())else'complete',UTC=utc(),expected=len(tasks()),completed=completed,failed_attempts=len(list((R/'runs').glob('*/attempt*/FAILURE.json'))),clock_sha256=sha(R/'CLOCK.json'),no_fitted_weights=True,independent_audit_available=False),R/'CAMPAIGN_COMPLETION.json')
 dump(load('CAMPAIGN_COMPLETION.json'),R/'LIVE_STATE.json')
except BaseException:
 (R/'logs/COORDINATOR_FAILURE.txt').write_text(traceback.format_exc());dump(dict(stage='needs_execution_review',UTC=utc(),private_diagnostic='logs/COORDINATOR_FAILURE.txt',clock_sha256=sha(R/'CLOCK.json')),R/'LIVE_STATE.json');raise
