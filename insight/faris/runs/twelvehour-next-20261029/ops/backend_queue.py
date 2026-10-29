"""Autonomous owner queue, predeclared pilots, strict gates, immutable deadline."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,subprocess,sys,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import check_seal,load,dump,sha,utc,tasks,catalog
a=argparse.ArgumentParser();a.add_argument('--owner',required=True);owner=a.parse_args().owner
c=check_seal();p=load('COLAB_RESOURCE_ADDENDUM.json')['assignments'][owner];all_tasks=tasks();slots=p['slots'];running={};finished=[]
def invoke(task,pilot=False):
    d=R/f'runs/{task}/attempt1'
    if d.exists():return
    log=R/'logs'/f'{task}.log';log.parent.mkdir(exist_ok=True);stream=log.open('a')
    process=subprocess.Popen([sys.executable,'-u',str(R/'ops/backend_worker.py'),'--owner',owner,'--task',task]+(['--pilot']if pilot else[]),stdout=stream,stderr=subprocess.STDOUT);stream.close();return process
started=time.time();pilot_times=[]
for task in p['pilot_tasks']:
    q=invoke(task,True)
    if q is not None:
        timeout=min(1850,max(1,c['model_deadline_epoch']-time.time()-30))
        try:q.wait(timeout=timeout)
        except subprocess.TimeoutExpired:q.terminate();q.wait(timeout=30)
    d=R/f'runs/{task}/attempt1'
    if not(d/'RESULT.json').exists():
        dump(dict(status='FAIL_EXECUTION',UTC=utc(),owner=owner,failed_pilot=task,execution_failure_not_predictive=True),R/'BACKEND_GATE.json');raise RuntimeError('Registered owner pilot failed; no broad admission')
    result=load(d/'RESULT.json');assert result['status']=='complete'and sha(d/'prediction.npz')==result['prediction_sha256'];pilot_times.append(result['seconds'])
duration=max(pilot_times);remaining=p['planned_fits']-len(p['pilot_tasks']);fits=duration<=1800 and time.time()+duration*remaining*1.5/slots+600<c['model_deadline_epoch']
gate=dict(status='PASS'if fits else'FAIL_RESOURCE_FEASIBILITY',UTC=utc(),owner=owner,pilot_seconds=pilot_times,max_pilot_seconds=duration,safety_factor=1.5,planned_owner_fits=p['planned_fits'],source_seal_sha256=sha(R/'SOURCE_SEAL.json'),clock_sha256=sha(R/'CLOCK.json'),contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),config_sha256=sha(R/'CONFIG.json'),score_used_for_admission=False,resource_addendum_sha256=sha(R/'COLAB_RESOURCE_ADDENDUM.json'))
dump(gate,R/'BACKEND_GATE.json')
if not fits:raise RuntimeError('Resource feasibility gate failed; no broad admission')
groups={catalog()[d]['admission_group']for d in p['designs']}
for group in groups:dump(dict(gate,group=group),R/('GATE_FAMILY_'+group+'.json'))
pending=[t for d in p['designs']for t,s in all_tasks.items()if s['design']==d and not(R/f'runs/{t}/attempt1').exists()]
dump(dict(UTC=utc(),owner=owner,original_deadline_epoch=c['deadline_epoch'],starting_actual_training=True),R/'VM_STARTED_TRAINING.json')
while pending or running:
    while pending and len(running)<slots and time.time()<c['admission_end_epoch'] and time.time()+duration*1.5+30<c['model_deadline_epoch']:
        task=pending.pop(0);q=invoke(task)
        if q is not None:running[task]=q
    for task,q in list(running.items()):
        if q.poll()is not None:finished.append(dict(task=task,returncode=q.returncode));del running[task]
    dump(dict(UTC=utc(),owner=owner,running={t:q.pid for t,q in running.items()},pending=len(pending),finished=finished,completed=len(list((R/'runs').glob('*/attempt1/RESULT.json'))),deadline_epoch=c['deadline_epoch']),R/'BACKEND_STATE.json')
    if time.time()>=c['model_deadline_epoch']-10:
        for q in running.values():q.terminate()
        break
    if time.time()>=c['admission_end_epoch']and not running:break
    if not running and pending and time.time()+duration*1.5+30>=c['model_deadline_epoch']:break
    time.sleep(3)
dump(dict(UTC=utc(),owner=owner,status='terminal',finished=finished,pending=pending,interrupted=list(running),clock_sha256=sha(R/'CLOCK.json'),source_unchanged=True),R/'BACKEND_FINAL.json')
