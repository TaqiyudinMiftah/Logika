"""Bounded lab queue. Run only as a fresh campaign-owned systemd user unit.

EXECUTION_PLAN.json pools maps pool to slots/resource/designs/pilot_tasks/python,
memory_max, gpu_indices. CPU threads fixed2. Whole design ownership is immutable.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse, fcntl, json, subprocess, sys, time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import check_seal,load,dump,utc,sha,tasks,catalog

def idle(index):
    q=subprocess.run(['nvidia-smi',f'--id={index}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if q.returncode:return False
    m,u=map(lambda x:int(x.strip()),q.stdout.strip().split(','));return m<=512 and u<=5

def run(pool):
    lock=(R/f'queue_{pool}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    clock=check_seal();plan=load('EXECUTION_PLAN.json');assert sha(R/'EXECUTION_PLAN.json')==clock['execution_plan_sha256']
    p=plan['pools'][pool];registered=tasks();designs=catalog();owned=set(p['designs']);pilots=list(p['pilot_tasks'])
    assert owned and owned<=set(designs)
    assert set(pilots)<=set(load('CONFIG.json')['pilot_tasks'])
    assert all(registered[t]['design'] in owned for t in pilots)
    assert all(designs[d]['resource']==p['resource'] for d in owned)
    for name,other in plan['pools'].items():
        if name!=pool:assert not owned&set(other['designs']),'Duplicate whole-design ownership'
    pending=pilots+[t for t,v in registered.items() if v['design'] in owned and t not in pilots]
    running={};finished=[];retired={};started=set()
    prefix=plan['unit_prefix']+'-'+pool
    def dispatch(task,index,pilot):
        unit=prefix+'-'+str(len(started))
        log=R/'logs'/f'{task}.log';log.parent.mkdir(exist_ok=True)
        duration=min(float(designs[registered[task]['design']].get('max_seconds',1800))+45,
                     clock['model_deadline_epoch']-time.time()-10)
        cmd=['systemd-run','--user','--quiet','--unit='+unit,'--property=MemoryMax='+p['memory_max'],
             '--property=CPUQuota=200%','--property=KillMode=control-group','--property=RuntimeMaxSec='+str(max(1,int(duration))),
             '--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log)]
        if index is not None:cmd+=['--setenv=CUDA_VISIBLE_DEVICES='+str(index)]
        cmd+=[p['python'],'-u',str(R/'source/worker.py'),'--task',task]+(['--pilot'] if pilot else [])
        result=subprocess.run(cmd,capture_output=True,text=True)
        record=dict(UTC=utc(),task=task,unit=unit,index=index,pilot=pilot,dispatch_returncode=result.returncode)
        with(R/f'DISPATCH_{pool}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        started.add(task)
        if result.returncode:
            record['stderr_sha256']=__import__('hashlib').sha256(result.stderr.encode()).hexdigest()
            retired[task]='dispatch_failed_no_automatic_retry';finished.append(record);return
        running[task]=record
    while time.time()<clock['model_deadline_epoch']-10:
        for task,info in list(running.items()):
            s=subprocess.run(['systemctl','--user','show',info['unit'],'-p','ActiveState','--value'],capture_output=True,text=True).stdout.strip()
            if s in ['active','activating','reloading']:continue
            folder=R/f'runs/{task}/attempt1'
            terminal=any((folder/name).exists() for name in ['RESULT.json','FAILURE.json','INTERRUPTION.json'])
            if not terminal:
                folder.mkdir(parents=True,exist_ok=True)
                dump(dict(status='interrupted',UTC=utc(),task=task,unit=info['unit'],reason='Owned unit ended without terminal receipt',no_automatic_retry=True,scientific_conclusion=False),folder/'INTERRUPTION.json')
            finished.append(dict(info,state=s,complete=(folder/'RESULT.json').exists()));del running[task]
        if time.time()<clock['admission_end_epoch']:
            for task in pending:
                if len(running)>=p['slots']:break
                if task in started or task in retired:continue
                folder=R/f'runs/{task}/attempt1'
                if folder.exists():retired[task]='existing_attempt_preserved';continue
                spec=designs[registered[task]['design']];pilot=task in pilots;gate=R/f"GATE_FAMILY_{spec['admission_group']}.json"
                if not pilot:
                    if not gate.exists():continue
                    g=json.loads(gate.read_text())
                    if g['status']!='PASS':retired[task]='family_gate_'+g['status'];continue
                    bound=g['max_pilot_seconds']*1.5
                else:bound=float(spec.get('max_seconds',1800))
                if time.time()+min(1800,bound)+30>=clock['model_deadline_epoch']:
                    retired[task]='conservative_model_deadline';continue
                index=None
                if p['resource']=='gpu':
                    occupied={v['index'] for v in running.values()}
                    available=[i for i in p['gpu_indices'] if i not in occupied and idle(i)]
                    if not available:continue
                    index=available[0]
                dispatch(task,index,pilot)
        dump(dict(UTC=utc(),pool=pool,running=running,finished=finished,retired=retired,
                  pending=len([t for t in pending if t not in started and t not in retired]),clock_sha256=sha(R/'CLOCK.json')),R/f'QUEUE_{pool}_STATE.json')
        if not running and (time.time()>=clock['admission_end_epoch'] or all(t in started or t in retired for t in pending)):break
        time.sleep(5)
    for task,info in running.items():
        subprocess.run(['systemctl','--user','stop',info['unit']],capture_output=True)
        folder=R/f'runs/{task}/attempt1';folder.mkdir(parents=True,exist_ok=True)
        if not any((folder/n).exists() for n in ['RESULT.json','FAILURE.json','INTERRUPTION.json']):
            dump(dict(status='interrupted',UTC=utc(),reason='Original model deadline',task=task,no_automatic_retry=True,scientific_conclusion=False),folder/'INTERRUPTION.json')
    dump(dict(UTC=utc(),status='terminal',pool=pool,finished=finished,interrupted=list(running),
              not_admitted=[t for t in pending if t not in started],retired=retired,clock_sha256=sha(R/'CLOCK.json')),R/f'QUEUE_{pool}_FINAL.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pool',required=True);run(a.parse_args().pool)
