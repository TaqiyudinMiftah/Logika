"""Bounded, reconciled CPU/GPU scheduling on the two user-authorized lab nodes."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import numpy as np
from common import R,load,dump,sha,utc,check_seal
from compose import completed


def state(stage,**fields):
    value=dict(stage=stage,UTC=utc(),**fields);dump(value,R/'LIVE_STATE.json')
    print(json.dumps(value),flush=True)


def command(node,args,timeout=60,check=True):
    return subprocess.run(args if node.get('local')else node['ssh']+[shlex.join(args)],
                          capture_output=True,text=True,timeout=timeout,check=check)


def resources(node):
    code="from pathlib import Path;import json,os,subprocess,shutil;m=dict(l.split(':',1)for l in Path('/proc/meminfo').read_text().splitlines());g=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);print(json.dumps({'cores':os.cpu_count(),'RAM_GiB':int(m['MemAvailable'].split()[0])/2**20,'disk_GiB':shutil.disk_usage(Path.home()).free/2**30,'GPU':[list(map(int,l.split(',')))for l in g.strip().splitlines()]}))"
    return json.loads(command(node,['python3','-c',code],25).stdout)


def rsync(node,source,destination):
    subprocess.run(['rsync','-az','-e',shlex.join(node['ssh'][:-1]),source,destination],check=True,timeout=300)


def deploy(node):
    if node.get('local'):return
    command(node,['mkdir','-p',node['root']])
    rsync(node,str(R/'source')+'/',node['ssh'][-1]+':'+node['root']+'/source/')
    rsync(node,str(R/'packets')+'/',node['ssh'][-1]+':'+node['root']+'/packets/')
    names=['CONFIG.json','VALIDATION_CONTRACT.json','CLOCK.json','PATHS.json','SOURCE_SEAL.json','GATE.json','PREPARATION_AUDIT.json']
    names+=list(load('VALIDATION_CONTRACT.json')['immutable_input_metadata'])
    for name in set(names):
        if(R/name).exists():rsync(node,str(R/name),node['ssh'][-1]+':'+node['root']+'/'+name)


def start(node,task_id,slot,pilot=False):
    config=load('CONFIG.json');task=config['tasks'][task_id];clock=check_seal();family=task['family']
    attempts=len(list((R/'dispatch'/task_id).glob('attempt*.json')))+1
    assert attempts<=2
    unit=f'logika-moe28-{task_id}-a{attempts}'.replace('__','-')
    limit=min(1800,int(clock['deadline_epoch']-time.time()-3600));assert limit>0
    args=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit='+unit,
          '-p','WorkingDirectory='+node['root'],'-p','CPUQuota=200%',
          '-p','MemoryMax='+('18G'if family=='kumo'else'8G'),'-p','MemorySwapMax=0',
          '-p','RuntimeMaxSec='+str(limit),'--setenv=CUDA_VISIBLE_DEVICES='+(str(slot)if family=='kumo'else''),
          '--setenv=HF_HOME='+node['HF_HOME'],node['python'],node['root']+'/source/worker.py',
          '--task',task_id,'--attempt',str(attempts)]
    if pilot:args.append('--pilot')
    (R/'logs').mkdir(exist_ok=True);log=(R/'logs'/f'{task_id}_a{attempts}.log').open('ab')
    process=subprocess.Popen(args if node.get('local')else node['ssh']+[shlex.join(args)],stdout=log,stderr=subprocess.STDOUT)
    record=dict(task_id=task_id,attempt=attempts,node=node,slot=slot,unit=unit,pilot=pilot,
                start_epoch=time.time(),runtime_limit=limit)
    dump(record,R/f'dispatch/{task_id}/attempt{attempts}.json')
    record.update(process=process,log=log);return record


def probe(job):
    node=job['node'];p=node['root']+f"/runs/{job['task_id']}/attempt{job['attempt']}"
    code="from pathlib import Path;import json,subprocess;p=Path("+repr(p)+");s=subprocess.run(['systemctl','--user','show',"+repr(job['unit'])+",'-p','ActiveState','--value'],capture_output=True,text=True);print(json.dumps({'active':s.stdout.strip()in['active','activating'],'result':(p/'RESULT.json').exists(),'failure':(p/'FAILURE.json').exists()}))"
    return json.loads(command(node,['python3','-c',code],25).stdout)


def collect(job):
    node=job['node'];relative=f"runs/{job['task_id']}/attempt{job['attempt']}";root=R/relative
    if not node.get('local'):
        root.mkdir(parents=True,exist_ok=True)
        rsync(node,node['ssh'][-1]+':'+node['root']+'/'+relative+'/',str(root)+'/')
    if(root/'RESULT.json').exists():
        record=json.loads((root/'RESULT.json').read_text());z=np.load(root/'prediction.npz')
        assert sha(root/'prediction.npz')==record['prediction_sha256']
        assert z['raw'].shape==(20000,)and np.isfinite(z['raw']).all()
        return True
    return False


def reconcile(job):
    status=probe(job)
    if status['active']:return None
    result=collect(job)if status['result']or status['failure']else False
    if job.get('log')is not None:job['log'].close()
    return result


def fit_pilots(nodes):
    config=load('CONFIG.json');clock=check_seal()
    for task_id in config['pilot_tasks']:
        if completed(task_id)is not None:continue
        family=config['tasks'][task_id]['family'];chosen=None
        while chosen is None:
            assert time.time()<min(clock['admission_end_epoch'],clock['started_epoch']+10800),'No pilot capacity within qualification allowance'
            for node in nodes:
                info=resources(node)
                if info['RAM_GiB']<20 or info['disk_GiB']<3:continue
                if family!='kumo':chosen=(node,-1);break
                for gpu,used,util in info['GPU']:
                    if used<256 and util<10:chosen=(node,gpu);break
                if chosen:break
            if chosen is None:state('waiting_for_pilot_capacity',task=task_id);time.sleep(20)
        job=start(chosen[0],task_id,chosen[1],pilot=True)
        state('runtime_pilot',task=task_id,node=chosen[0]['name'],slot=chosen[1])
        while True:
            outcome=reconcile(job)
            if outcome is not None:break
            time.sleep(10)
        assert outcome,('Runtime pilot failed',task_id)


def main(resume=False):
    lock=(R/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    clock=check_seal();nodes=load('TOPOLOGY.json')['nodes'];config=load('CONFIG.json')
    assert {n['name']for n in nodes}=={'node2001','node2003'}
    if not(R/'PREPARATION_AUDIT.json').exists():
        if resume:
            previous=subprocess.run(['systemctl','--user','show','logika-moe28-prepare','-p','ActiveState','--value'],capture_output=True,text=True)
            assert previous.stdout.strip()not in ['active','activating'],'Existing preparation unit is still active; reconcile it first.'
            state('reconciled_preparation_resume',original_clock_sha256=sha(R/'CLOCK.json'))
        state('preparing_features_and_anchor_crossfit',clock=clock)
        args=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit=logika-moe28-prepare',
              '-p','WorkingDirectory='+str(R),'-p','CPUQuota=400%','-p','MemoryMax=22G','-p','MemorySwapMax=0',
              '-p','RuntimeMaxSec=7200',sys.executable,str(R/'source/prepare.py')]
        with(R/'preparation.log').open('ab')as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7260)
    for node in nodes:deploy(node)
    if not(R/'GATE.json').exists():
        fit_pilots(nodes)
        subprocess.run([sys.executable,str(R/'source/gate.py')],check=True,timeout=600)
        for node in nodes:deploy(node)
    assert load('GATE.json')['status']=='PASS'
    completed_tasks={t for t in config['tasks']if completed(t)is not None};active=[];failed=[];pending=[]
    for task_id in config['tasks']:
        if task_id in completed_tasks:continue
        prior=list((R/'dispatch'/task_id).glob('attempt*.json'))
        if prior:
            assert resume,'Existing dispatch unexpectedly found; reconcile through --resume'
            record=json.loads(sorted(prior)[-1].read_text());record.update(process=None,log=None)
            result=reconcile(record)
            if result is None:active.append(record)
            elif result:completed_tasks.add(task_id)
            else:failed.append(dict(task_id=task_id,reason='Prior attempt failed; no automatic resource retry.'))
        else:pending.append(task_id)
    last=0;errors={}
    while pending or active:
        infos={}
        for node in nodes:
            try:infos[node['name']]=resources(node);errors.pop(node['name'],None)
            except(Exception,)as error:errors[node['name']]=str(error)
        for job in list(active):
            if job['node']['name']not in infos:continue
            try:outcome=reconcile(job)
            except(Exception,)as error:errors[job['node']['name']]=str(error);continue
            if outcome is not None:
                active.remove(job)
                if outcome:completed_tasks.add(job['task_id'])
                else:failed.append(dict(task_id=job['task_id'],attempt=job['attempt'],reason='Recorded execution failure; no model-quality conclusion.'))
        stop=(R/'STOP').exists()or time.time()>=clock['admission_end_epoch']
        if not stop:
            for node in nodes:
                info=infos.get(node['name'])
                if info is None or info['disk_GiB']<3:continue
                owned=[j for j in active if j['node']['name']==node['name']]
                free=info['RAM_GiB']-sum(18 if config['tasks'][j['task_id']]['family']=='kumo' else 8 for j in owned)
                gpu_owned={j['slot']for j in owned if config['tasks'][j['task_id']]['family']=='kumo'}
                for gpu,used,util in info['GPU']:
                    index=next((i for i,t in enumerate(pending)if config['tasks'][t]['family']=='kumo'),None)
                    if index is not None and gpu not in gpu_owned and used<256 and util<10 and free>=20:
                        job=start(node,pending.pop(index),gpu);active.append(job);owned.append(job);gpu_owned.add(gpu);free-=18
                busy_gpu=sum(used>=256 or util>=10 for _,used,util in info['GPU'])
                capacity=max(0,min(3,(info['cores']-2-2*max(busy_gpu,len(gpu_owned)))//2))
                cpu_owned=sum(config['tasks'][j['task_id']]['family']!='kumo'for j in owned)
                while free>=10 and cpu_owned<capacity:
                    index=next((i for i,t in enumerate(pending)if config['tasks'][t]['family']!='kumo'),None)
                    if index is None:break
                    active.append(start(node,pending.pop(index),cpu_owned));cpu_owned+=1;free-=8
        if time.time()-last>30:
            state('modelling',completed=len(completed_tasks),expected=len(config['tasks']),failed=len(failed),pending=len(pending),
                  active=[{k:j[k]for k in ['task_id','attempt','slot','start_epoch']}|dict(node=j['node']['name'])for j in active],
                  failures=failed,node_errors=errors,clock=clock);last=time.time()
        if stop and not active:break
        if time.time()>clock['deadline_epoch']-7200:
            for job in active:command(job['node'],['systemctl','--user','stop',job['unit']],30,False)
            break
        time.sleep(15)
    dump(dict(completed=sorted(completed_tasks),failed=failed,pending=pending,
              unresolved=[j['task_id']for j in active],UTC=utc()),R/'MODEL_RUN_AUDIT.json')
    state('fitting_combiners',completed=len(completed_tasks),failed=len(failed),pending=len(pending))
    subprocess.run([sys.executable,str(R/'source/compose.py')],check=True,timeout=1800)
    state('terminal_evaluation',frozen_sha256=sha(R/'FROZEN.json'))
    subprocess.run([sys.executable,str(R/'source/evaluate.py')],check=True,timeout=3600)
    result=load('evaluation/RESULT.json')
    state('refit_required'if result['primary_development_gain_supported']else'complete',
          primary=result['primary_result'],supported=result['primary_development_gain_supported'],
          local_target_054_point_reached=result['local_target_054_point_reached'],
          incomplete_cohorts=result['incomplete_cohorts'],deadline_epoch=clock['deadline_epoch'],submission_made=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    try:main(a.resume)
    except BaseException:state('failed',error=traceback.format_exc());raise
