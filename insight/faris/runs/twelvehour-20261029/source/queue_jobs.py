"""Bounded execution only. Qualification never uses candidate scores."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,json,subprocess,sys,time
from common import R,load,dump,utc,check_seal,tasks,catalog

def gpu_idle(index):
    p=subprocess.run(['nvidia-smi',f'--id={index}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if p.returncode:return False
    memory,util=[int(v.strip())for v in p.stdout.strip().split(',')];return memory<=512 and util<=5

def main(pool):
    c=check_seal();cfg=load('CONFIG.json');all_tasks=tasks();designs=catalog();slots=2 if pool in ['cpu2001','gpu2001']else 3
    running={};finished=[];started=set();retired=set();stopped=False
    # CPU designs split2:3 between nodes; pilots are the sole explicit ownership exception.
    cpu_designs=[d for group in cfg['priority_groups']for d,s in designs.items()if s['resource']=='cpu' and s['admission_group']==group]
    owned=set(cpu_designs[i]for i in range(len(cpu_designs))if (i%5<2)==(pool=='cpu2001'))
    if pool=='gpu2001':owned={d for d,s in designs.items()if s['resource']=='gpu'}
    pilot=[t for t in cfg['pilot_tasks']if (designs[all_tasks[t]['design']]['resource']=='gpu')==(pool=='gpu2001')] if pool!='cpu2003'else []
    ordinary=[t for d in designs if d in owned for t,v in all_tasks.items()if v['design']==d and t not in cfg['pilot_tasks']]
    pending=pilot+list(t for t in ordinary if t not in pilot)
    while time.time()<c['model_deadline_epoch']-10:
        for task,info in list(running.items()):
            unit=info['unit'];state=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],capture_output=True,text=True).stdout.strip()
            if state not in ['active','activating','reloading']:
                folder=R/f'runs/{task}/attempt1';record=dict(task=task,unit=unit,state=state,completed=(folder/'RESULT.json').exists(),failed=(folder/'FAILURE.json').exists())
                if not record['completed'] and not record['failed']:dump(dict(status='interrupted',UTC=utc(),task=task,reason='Owned worker unit ended without terminal receipt',no_automatic_retry=True),folder/'INTERRUPTION.json')
                finished.append(record);del running[task]
        if time.time()>=c['admission_end_epoch']:
            if not running:break
            time.sleep(5);continue
        for task in pending:
            if len(running)>=slots:break
            if task in started or task in retired:continue
            folder=R/f'runs/{task}/attempt1';spec=designs[all_tasks[task]['design']];group=spec['admission_group'];is_pilot=task in pilot
            if folder.exists():retired.add(task);continue
            if not is_pilot:
                gate=R/('GATE_FAMILY_'+group+'.json')
                if not gate.exists():continue
                if json.loads(gate.read_text())['status']!='PASS':retired.add(task);continue
            index=None
            if pool=='gpu2001':
                occupied={x['index']for x in running.values()};idle=[i for i in range(2)if i not in occupied and gpu_idle(i)]
                if not idle:continue
                index=idle[0]
            # Predict conservative finish from a registered per-fit cap (pilots) or measured gate.
            gate=R/('GATE_FAMILY_'+group+'.json')
            bound=float(json.loads(gate.read_text()).get('max_pilot_seconds',1800))*1.5 if gate.exists()else float(spec['max_seconds'])
            if time.time()+min(1800,bound)+20>=c['model_deadline_epoch']:retired.add(task);continue
            unit='logika-twelve-'+pool+'-'+str(len(started))+'-20261029'
            command=['systemd-run','--user','--quiet','--unit='+unit,'--property=MemoryMax='+('12G'if pool=='gpu2001'else'10G'),'--property=CPUQuota=200%','--property=RuntimeMaxSec='+str(min(1850,int(c['model_deadline_epoch']-time.time()-10))),'--property=StandardOutput=append:'+str(R/'logs'/f'{task}.log'),'--property=StandardError=append:'+str(R/'logs'/f'{task}.log')]
            if index is not None:command+=['--setenv=CUDA_VISIBLE_DEVICES='+str(index)]
            py='/mnt/home/faris/logika-model-arena-20261025/env311/bin/python'if pool=='gpu2001'else'/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
            command +=[py,'-u',str(R/'source/worker.py'),'--task',task]+(['--pilot']if is_pilot else [])
            (R/'logs').mkdir(exist_ok=True);subprocess.run(command,check=True,capture_output=True,text=True)
            running[task]=dict(unit=unit,index=index,pilot=is_pilot);started.add(task)
            with (R/f'DISPATCH_{pool}.jsonl').open('a')as f:f.write(json.dumps(dict(UTC=utc(),task=task,**running[task]))+'\n')
        dump(dict(UTC=utc(),pool=pool,started=len(started),finished=finished,running=running,pending=len([t for t in pending if t not in started|retired]),source_clock_unchanged=True),R/f'QUEUE_{pool}_STATE.json')
        if all(t in started|retired for t in pending) and not running:break
        time.sleep(5)
    for task,info in running.items():subprocess.run(['systemctl','--user','stop',info['unit']],capture_output=True)
    dump(dict(UTC=utc(),status='terminal',pool=pool,finished=finished,interrupted=list(running),not_admitted=[t for t in pending if t not in started],clock_preserved=True),R/f'QUEUE_{pool}_FINAL.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pool',choices=['cpu2001','cpu2003','gpu2001'],required=True);main(a.parse_args().pool)
