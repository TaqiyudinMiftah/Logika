"""Execute a pre-recorded prefix with measured gates and no automatic fit retries."""
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,time,hashlib
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args()
planpath=Path(args.plan);plan=json.loads(planpath.read_text());R=planpath.parent.parent
C=json.loads((R/'CLOCK.json').read_text());D=R/'ownership'/plan['owner'];D.mkdir(exist_ok=True)
def dump(path,value):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(path)
def pair(task):
    start=time.time();ident=f"{task['kind']}__s{task['seed']}__f{task['fold']}"
    for phase in ['validation','production']:
        out=R/'runs'/ident/phase
        if out.exists():return dict(task=task,status='existing_attempt_not_retried',seconds=time.time()-start)
        if time.time()>=C['admission_end_epoch']:return dict(task=task,status='not_admitted_clock',seconds=time.time()-start)
        cmd=[plan['python'],'-u',str(R/'worker.py'),'--kind',task['kind'],'--seed',str(task['seed']),'--fold',str(task['fold']),'--phase',phase]
        log=R/'logs'/(ident+'_'+phase+'.log');log.parent.mkdir(exist_ok=True)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(plan.get('gpu',''));env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        with log.open('ab') as f:
            try:q=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,C['model_deadline_epoch']-time.time()-15))
            except subprocess.TimeoutExpired:
                out.mkdir(exist_ok=True,parents=True);dump(out/'INTERRUPTION.json',dict(reason='owned_model_deadline',epoch=time.time(),no_retry=True));return dict(task=task,status='deadline_interruption',seconds=time.time()-start)
        if q.returncode:return dict(task=task,status='failure_not_retried',seconds=time.time()-start)
    return dict(task=task,status='complete_pair',seconds=time.time()-start)
wait=plan.get('wait_unit')
while wait and time.time()<C['admission_end_epoch']:
    if subprocess.run(['systemctl','--user','is-active','--quiet',wait]).returncode:break
    time.sleep(3)
assert json.loads((R/'GATE.json').read_text())['status']=='PASS'
dump(D/'START.json',dict(epoch=time.time(),owner=plan['owner'],plan_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),clock_sha256=hashlib.sha256((R/'CLOCK.json').read_bytes()).hexdigest(),gate_sha256=hashlib.sha256((R/'GATE.json').read_bytes()).hexdigest()))
groups={}
for task in plan['tasks']:groups.setdefault(task['kind'],[]).append(task)
results=[];pilots={}
with concurrent.futures.ThreadPoolExecutor(max_workers=plan['slots']) as ex:
    fs={ex.submit(pair,ts[0]):k for k,ts in groups.items()}
    for f in concurrent.futures.as_completed(fs):
        v=f.result();pilots[fs[f]]=v;results.append(v);dump(D/'STATE.json',dict(epoch=time.time(),results=results))
remaining=[t for k,ts in groups.items() if pilots[k]['status']=='complete_pair' for t in ts[1:]]
work=sum(pilots[k]['seconds']*(len(ts)-1) for k,ts in groups.items() if pilots[k]['status']=='complete_pair')
estimate=work*1.5/max(1,plan['slots'])+300
passed=all(v['status']=='complete_pair' for v in pilots.values()) and time.time()+estimate<C['model_deadline_epoch'] and time.time()<C['admission_end_epoch']
dump(D/'RUNTIME_GATE.json',dict(status='PASS' if passed else 'FAIL_RUNTIME_OR_EXECUTION',epoch=time.time(),pilots=pilots,remaining_pairs=len(remaining),conservative_remaining_seconds=estimate,model_deadline_epoch=C['model_deadline_epoch'],each_phase_still_requires_admission_clock=True,scientific_negative=False))
if passed:
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan['slots']) as ex:
        fs=[ex.submit(pair,t) for t in remaining]
        for f in concurrent.futures.as_completed(fs):results.append(f.result());dump(D/'STATE.json',dict(epoch=time.time(),results=results))
dump(D/'FINAL.json',dict(epoch=time.time(),owner=plan['owner'],results=results,planned_pairs=len(plan['tasks']),no_automatic_retry=True))
