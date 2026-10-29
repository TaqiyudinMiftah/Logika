"""Remote autonomous coordinator with one original twelve-hour clock."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,fcntl,json,subprocess,sys,time,traceback
from common import R,load,dump,sha,utc,check_seal,tasks,catalog

PY='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
HOST='faris@gpu-labkc.ub.ac.id';REMOTE=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','2003',HOST]

def sync_back():
    p=subprocess.run(['rsync','-az','--ignore-existing','-e','ssh -o BatchMode=yes -p2003',HOST+':'+str(R)+'/runs/',str(R/'runs')+'/'],capture_output=True,text=True,timeout=120)
    if p.returncode:dump(dict(UTC=utc(),returncode=p.returncode,private_message_hash=__import__('hashlib').sha256(p.stderr.encode()).hexdigest(),canonical_existing_results_preserved=True),R/'SYNC_RETRY_STATE.json')
    subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -p2003',HOST+':'+str(R)+'/QUEUE_cpu2003_FINAL.json',str(R)+'/'],capture_output=True,timeout=30)

def qualify():
    cfg=load('CONFIG.json');clock=check_seal();eligible=[]
    for group in cfg['priority_groups']:
        gate=R/('GATE_FAMILY_'+group+'.json')
        if gate.exists():continue
        pilots=cfg['group_pilots'][group];results=[];failed=[]
        for task in pilots:
            d=R/f'runs/{task}/attempt1'
            if (d/'RESULT.json').exists():results.append(load(d/'RESULT.json'))
            elif (d/'FAILURE.json').exists()or(d/'INTERRUPTION.json').exists():failed.append(task)
        if not failed and len(results)!=len(pilots):continue
        count=sum(1 for s in catalog().values()if s['admission_group']==group)*10
        duration=max([r['seconds']for r in results]or[1800]);slots=2 if group.startswith('gpu')else 5
        existing_work=0.
        for other in cfg['priority_groups']:
            q=R/('GATE_FAMILY_'+other+'.json')
            if other==group or not q.exists()or other.startswith('gpu')!=group.startswith('gpu'):continue
            g=load(q)
            if g['status']!='PASS':continue
            unfinished=sum(1 for task,v in tasks().items()if catalog()[v['design']]['admission_group']==other and not (R/f'runs/{task}/attempt1/RESULT.json').exists() and not (R/f'runs/{task}/attempt1/FAILURE.json').exists())
            existing_work+=g['max_pilot_seconds']*unfinished*cfg['admission_safety_factor']
        projected=existing_work+duration*max(0,count-len(results))*cfg['admission_safety_factor']
        supported=not failed and duration<=cfg['group_pilot_max_seconds'] and time.time()+projected/slots+cfg['admission_finish_margin']<clock['model_deadline_epoch']
        status='FAIL_EXECUTION'if failed else('PASS'if supported else'FAIL_RESOURCE_FEASIBILITY')
        for task in pilots:
            p=R/f'runs/{task}/attempt1'
            if(p/'RESULT.json').exists():assert sha(p/'prediction.npz')==load(p/'RESULT.json')['prediction_sha256']
        result=dict(status=status,UTC=utc(),group=group,pilot_tasks=pilots,pilot_seconds=[r['seconds']for r in results],failed_pilots=failed,max_pilot_seconds=duration,planned_group_fits=count,projected_seconds_including_other_admitted_work=projected/slots,projection_slots=slots,safety_factor=cfg['admission_safety_factor'],finish_margin_seconds=cfg['admission_finish_margin'],score_used_for_admission=False,clock_sha256=sha(R/'CLOCK.json'),contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),config_sha256=sha(R/'CONFIG.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),scientific_conclusion_from_execution_failure=False)
        dump(result,gate);eligible.append(gate)
    gates=list(R.glob('GATE_FAMILY_*.json'))
    if gates:subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -p2003',*[str(g)for g in gates],HOST+':'+str(R)+'/'],check=True,capture_output=True,timeout=120)

def run_module(name,extra=()):
    with (R/'logs'/f'{name}.log').open('a')as stream:subprocess.run([PY,str(R/'source'/f'{name}.py'),*extra],check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=max(30,int(load('CLOCK.json')['deadline_epoch']-time.time()-10)))

def start_unit(name,pool,remote=False):
    clock=load('CLOCK.json');remaining=max(30,int(clock['model_deadline_epoch']-time.time()))
    command=['systemd-run','--user','--unit='+name,'--property=MemoryMax=2G','--property=CPUQuota=100%','--property=RuntimeMaxSec='+str(remaining),PY,'-u',str(R/'source/queue_jobs.py'),'--pool',pool]
    subprocess.run((REMOTE if remote else[])+command,check=True,capture_output=True,text=True)

def launch():
    lock=(R/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not(R/'CLOCK.json').exists():
        receipt=load('PREPUBLICATION.json');assert len(receipt['remote_commit'])==40 and receipt['config_sha256']==sha(R/'CONFIG.json') and receipt['source_seal_sha256']==sha(R/'SOURCE_SEAL.json')
        # Skill initializer and the scientific clock are created together before first fitting.
        subprocess.run(['/usr/bin/python3',str(R/'ops/init_competition.py'),str(R),'logika-twelvehour-20261029','pooled absolute binary KS','higher','12','--train',str(R/'supplied_data/train_applications.csv'),'--test',str(R/'supplied_data/test_applications.csv'),'--compute','Lab2001 two RTX3060 GPUs and2CPUslots; lab2003 threeCPUslots; no paid runtime','--rules',str(R/'evidence/rules.md')],check=True,capture_output=True,text=True)
        state=load('TRAINING_STATE.json');deadline=state;start=__import__('datetime').datetime.fromisoformat(deadline['start_utc']).timestamp()
        cfg=load('CONFIG.json');dump(dict(started_epoch=start,started_UTC=deadline['start_utc'],admission_end_epoch=start+cfg['admission_hours']*3600,model_deadline_epoch=start+cfg['model_hours']*3600,deadline_epoch=start+12*3600,hours=12,config_sha256=sha(R/'CONFIG.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),skill_deadline_sha256=sha(R/'DEADLINE.md')),R/'CLOCK.json')
    clock=check_seal();assert time.time()<clock['deadline_epoch']
    if(R/'COMPLETION.json').exists():return
    if not(R/'GATE.json').exists():run_module('gate')
    assert load('GATE.json')['status']=='PASS'
    # Transfer no outer targets to model workers. Existing packet copies are hash-matched beforehand.
    subprocess.run(REMOTE+['mkdir','-p',str(R/'runs')],check=True,capture_output=True)
    for name in ['source','CONFIG.json','CLOCK.json','SOURCE_SEAL.json','VALIDATION_CONTRACT.json','INPUTS.json','GATE.json','ENVIRONMENT.json','CHECKPOINTS.json','evidence','TRAINING_BRIEF.md','DEADLINE.md','TRAINING_STATE.json','AGENTS.md']:
        subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -p2003',str(R/name),HOST+':'+str(R)+'/'],check=True,capture_output=True)
    start_unit('logika-twelvehour-cpu2001-20261029','cpu2001');start_unit('logika-twelvehour-cpu2003-20261029','cpu2003',True);start_unit('logika-twelvehour-gpu2001-20261029','gpu2001')
    last_evaluate=0
    while time.time()<clock['model_deadline_epoch']-90:
        sync_back();qualify()
        run_module('compose')
        if time.time()-last_evaluate>=300:run_module('evaluate');last_evaluate=time.time()
        completed=len(list((R/'runs').glob('*/attempt1/RESULT.json')));failed=len(list((R/'runs').glob('*/attempt1/FAILURE.json')));groups={p.stem:load(p)['status']for p in R.glob('GATE_FAMILY_*.json')}
        dump(dict(stage='running',UTC=utc(),completed_fits=completed,failed_attempts=failed,expected_fits=len(tasks()),family_gates=groups,clock_sha256=sha(R/'CLOCK.json'),laptop_required=False,no_paid_allocation=True),R/'LIVE_STATE.json')
        if all((R/f'QUEUE_{p}_FINAL.json').exists()for p in ['cpu2001','cpu2003','gpu2001']):break
        time.sleep(20)
    for name in ['logika-twelvehour-cpu2001-20261029','logika-twelvehour-gpu2001-20261029']:subprocess.run(['systemctl','--user','stop',name],capture_output=True)
    subprocess.run(REMOTE+['systemctl','--user','stop','logika-twelvehour-cpu2003-20261029'],capture_output=True)
    sync_back();run_module('compose');run_module('evaluate',['--final']);run_module('audit')
    completed=len(list((R/'runs').glob('*/attempt1/RESULT.json')));failed=len(list((R/'runs').glob('*/attempt1/FAILURE.json')))
    result=dict(stage='complete',UTC=utc(),completed_fits=completed,failed_attempts=failed,planned_fits=len(tasks()),audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json'),final_evaluation_sha256=sha(R/'FINAL_EVALUATION.json'),original_clock_sha256=sha(R/'CLOCK.json'),no_refit=True,no_submission=True,no_paid_allocation=True,weights_saved=False)
    dump(result,R/'COMPLETION.json');dump(result,R/'LIVE_STATE.json')

if __name__=='__main__':
    try:launch()
    except Exception:
        p=R/'private/COORDINATOR_FAILURE_PRIVATE.log';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc());dump(dict(UTC=utc(),status='needs_execution_review',private_trace_sha256=sha(p),clock_preserved=True,no_automatic_restart=True),R/'COORDINATOR_FAILURE.json');raise
