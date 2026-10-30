"""Continuous whole-design queue with measured, per-design admission.

No score is read. One full-size pilot belongs to each ten-fit cohort. Complete
cohorts take priority over another pilot; idle slots immediately take the next
predeclared hypothesis. No implicit retries, migration or recipe changes.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='2'
from pathlib import Path
import argparse,fcntl,json,signal,subprocess,sys,time,traceback,shutil
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import check_seal,load,dump,sha,utc,tasks,catalog

def idle(index):
    p=subprocess.run(['nvidia-smi',f'--id={index}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if p.returncode:return False
    memory,use=[int(v.strip())for v in p.stdout.strip().split(',')]
    return memory<=512 and use<=5

def main(owner):
    lock=(R/'owner_queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (R/'BACKEND_FINAL.json').exists():return
    clock=check_seal();plan=load('EXECUTION_PLAN.json');seal=load('DISPATCH_SEAL.json');assert sha(R/'EXECUTION_PLAN.json')==seal['execution_plan_sha256'] and sha(R/'CLOCK.json')==seal['clock_sha256']
    assignment=plan['owners'][owner];designs=catalog();alltasks=tasks();owned=assignment['designs']
    assert owned and all(d in designs for d in owned)
    other=[d for k,v in plan['owners'].items()if k!=owner for d in v['designs']];assert not set(owned)&set(other)
    assert load('GATE.json')['status']=='PASS' and load('BASELINE_REPLAY_GATE.json')['status']=='PASS'
    active={};states={d:'unstarted' for d in owned};durations={};consecutive_failures=0;dispatches=[]
    groups={designs[d]['admission_group']for d in owned};assert len(groups)==len(owned)
    expected={d:[t for t,v in alltasks.items()if v['design']==d]for d in owned}
    pilots={d:f'{d}__s2026__f0'for d in owned}
    def finished_path(task):return R/f'runs/{task}/attempt1'
    def terminal(task):return any((finished_path(task)/n).exists()for n in ['RESULT.json','FAILURE.json','INTERRUPTION.json'])
    def interrupt(task,why):
        d=finished_path(task);d.mkdir(parents=True,exist_ok=True)
        if not terminal(task):dump(dict(status='interrupted',UTC=utc(),task=task,reason=why,no_automatic_retry=True,scientific_conclusion=False),d/'INTERRUPTION.json')
    def gate(design,result=None,failure=None):
        group=designs[design]['admission_group'];path=R/('GATE_FAMILY_'+group+'.json')
        if path.exists():
            existing=load(path);durations[design]=float(existing['max_pilot_seconds']);states[design]='admitted' if existing['status']=='PASS' else existing['status'];return existing['status']=='PASS'
        duration=float(result['seconds'])if result else float(designs[design]['max_seconds'])
        pending_admitted=sum(sum(not terminal(t)for t in expected[d])*durations[d]*1.5 for d in owned if states[d]=='admitted')
        remaining=sum(not terminal(t)for t in expected[design]);projected=(pending_admitted+remaining*duration*1.5)/assignment['slots']
        fits=failure is None and duration<=designs[design]['max_seconds'] and time.time()+projected+300<clock['model_deadline_epoch']
        status='FAIL_EXECUTION'if failure else('PASS'if fits else'FAIL_RESOURCE_FEASIBILITY')
        record=dict(status=status,UTC=utc(),owner=owner,group=group,pilot_tasks=[pilots[design]],max_pilot_seconds=duration,pilot_seconds=[duration]if result else[],failed_pilots=[pilots[design]]if failure else[],planned_group_fits=10,projection_slots=assignment['slots'],projected_seconds=projected,safety_factor=1.5,finish_margin_seconds=300,score_used_for_admission=False,clock_sha256=sha(R/'CLOCK.json'),config_sha256=sha(R/'CONFIG.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),no_predictive_conclusion_from_failure=True)
        dump(record,path);durations[design]=duration
        states[design]='admitted'if fits else status
        return fits
    def start(task,pilot,index):
        d=finished_path(task)
        if d.exists():return False
        log=R/'logs'/f'{task}.log';log.parent.mkdir(exist_ok=True)
        env=dict(os.environ)
        if index is not None:env['CUDA_VISIBLE_DEVICES']=str(index)
        with log.open('ab')as stream:
            p=subprocess.Popen([sys.executable,'-u',str(R/'source/worker.py'),'--task',task]+(['--pilot']if pilot else[]),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        active[task]=dict(process=p,started=time.time(),index=index,pilot=pilot)
        dispatches.append(dict(task=task,pid=p.pid,UTC=utc(),pilot=pilot,index=index))
        dump(dict(owner=owner,dispatches=dispatches),R/'DISPATCH.json')
        if pilot:states[alltasks[task]['design']]='pilot_running'
        return True
    # On reboot/process failure preserve started attempts, never train one twice.
    previous_dispatch=load('DISPATCH.json').get('dispatches',[]) if (R/'DISPATCH.json').exists() else []
    for item in previous_dispatch:
        task=item['task'];folder=finished_path(task)
        if terminal(task) or not folder.exists():continue
        proc=Path('/proc')/str(item['pid'])/'cmdline'
        live=proc.read_bytes().replace(b'\x00',b' ').decode(errors='replace') if proc.exists() else ''
        if str(R/'source/worker.py') in live and task in live:
            raise RuntimeError('An actual prior owned worker is still active; reconcile before queue resume')
        interrupt(task,'Prior owned process ended or host rebooted; no automatic attempt retry')
    # Explicit resume preserves terminal outputs and never repeats existing attempts.
    for d in owned:
        p=finished_path(pilots[d])
        if(p/'RESULT.json').exists():gate(d,load(p/'RESULT.json'))
        elif p.exists():gate(d,failure='Existing pilot attempt; no automatic retry')
    while time.time()<clock['model_deadline_epoch']-20:
        for task,info in list(active.items()):
            process=info['process'];design=alltasks[task]['design']
            if process.poll() is None and time.time()-info['started']<float(designs[design]['max_seconds'])+45:continue
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            path=finished_path(task)
            if not terminal(task):interrupt(task,'Owned worker ended without a terminal receipt or exceeded per-fit execution bound')
            success=(path/'RESULT.json').exists()
            if success:
                v=load(path/'RESULT.json');assert sha(path/'START.json')==v['start_sha256']and sha(path/'prediction.npz')==v['prediction_sha256'];consecutive_failures=0
            else:consecutive_failures+=1
            if info['pilot']:gate(design,load(path/'RESULT.json')if success else None,None if success else 'Pilot execution failure')
            del active[task]
        if consecutive_failures>=5:
            dump(dict(status='SYSTEMIC_EXECUTION_FAILURE',UTC=utc(),owner=owner,consecutive_failures=consecutive_failures,no_more_admissions=True),R/'OWNER_REVIEW_REQUIRED.json');break
        for d in owned:
            if states[d]=='admitted'and all(terminal(t)for t in expected[d]):states[d]='complete_attempts'
        if time.time()<clock['admission_end_epoch']:
            while len(active)<assignment['slots']:
                if shutil.disk_usage(R).free<512*1024**2:
                    dump(dict(status='NEW_ADMISSION_PAUSED_STORAGE',UTC=utc(),owner=owner,minimum_free_bytes=512*1024**2,healthy_fits_retained=True),R/'RESOURCE_PAUSE.json');break
                task=None;pilot=False
                # Finish already admitted cohorts first.
                for d in owned:
                    if states[d]!='admitted':continue
                    for t in expected[d]:
                        if not finished_path(t).exists()and t not in active:task=t;break
                    if task:break
                if task is None:
                    d=next((d for d in owned if states[d]=='unstarted'),None)
                    if d is None:break
                    if time.time()+float(designs[d]['max_seconds'])+300>=clock['model_deadline_epoch']:states[d]='not_admitted_deadline';continue
                    task=pilots[d];pilot=True
                design=alltasks[task]['design']
                if not pilot and time.time()+durations[design]*1.5+30>=clock['model_deadline_epoch']:states[design]='not_admitted_deadline';continue
                index=None
                if assignment['resource']=='gpu':
                    occupied={v['index']for v in active.values()};indices=assignment.get('gpu_indices',[0])
                    allowed=[i for i in indices if i not in occupied and idle(i)]
                    if not allowed:break
                    index=allowed[0]
                if not start(task,pilot,index):states[design]='existing_attempt_preserved';continue
        pending=sum(states[d] in {'unstarted','pilot_running','admitted'}for d in owned)
        dump(dict(UTC=utc(),owner=owner,states=states,running={t:{k:v for k,v in a.items()if k!='process'}for t,a in active.items()},completed_fits=len(list((R/'runs').glob('*/attempt1/RESULT.json'))),pending_designs=pending,deadline_epoch=clock['deadline_epoch'],clock_sha256=sha(R/'CLOCK.json'),automatic_queue_refill=True),R/'BACKEND_STATE.json')
        if not active and (not pending or time.time()>=clock['admission_end_epoch']):break
        time.sleep(3)
    for task,info in active.items():
        p=info['process']
        if p.poll()is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        interrupt(task,'Original model deadline or systemic execution stop')
    dump(dict(status='terminal',UTC=utc(),owner=owner,states=states,completed_fits=len(list((R/'runs').glob('*/attempt1/RESULT.json'))),clock_sha256=sha(R/'CLOCK.json'),no_clock_extension=True,weights_saved=False),R/'BACKEND_FINAL.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--owner',required=True)
    try:main(a.parse_args().owner)
    except BaseException:
        p=R/'private/OWNER_FAILURE.log';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc());dump(dict(status='execution_review_required',UTC=utc(),private_trace_sha256=sha(p)),R/'OWNER_FAILURE.json');raise
