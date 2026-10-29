"""Explicit one-time attempt2 after hash-verified missing-packet staging repair.

Only pre-START failures qualify; science, seeds, data and clock are unchanged.
Original attempts move to immutable failure archives. Logical attempt1 links to
approved attempt2 only after its predictions and metadata pass full hash checks.
"""
from pathlib import Path
import hashlib,json,subprocess,time,datetime,shlex
R=Path('/mnt/home/faris/logika-twelvehour-20261029');HOST='faris@gpu-labkc.ub.ac.id';S=['ssh','-o','BatchMode=yes','-p2003',HOST]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(a):return subprocess.run(a,check=True,text=True,capture_output=True,timeout=180)
def remote(code):return json.loads(run(S+['python3 -c '+shlex.quote(code)]).stdout)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
clock=json.load(open(R/'CLOCK.json'))
code="""from pathlib import Path
import json,hashlib
r=Path(%r);out={}
for p in (r/'runs').glob('*/attempt1/FAILURE.json'):
 v=json.load(open(p));assert v['seconds']<2 and not (p.parent/'START.json').exists() and not(p.parent/'prediction.npz').exists();assert 'FileNotFoundError'in v['error'] and '/packets/fold'in v['error']
 out[p.parent.parent.name]={'original_failure_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'original_failure_UTC':v['UTC'],'approved_attempt':2,'reason':'Missingfold0-2staging, failedbeforeSTART/modeltraining; allfivepackethashesnowverified'}
print(json.dumps(out))
"""%str(R)
approved=remote(code);assert len(approved)==12
dump(R/'APPROVED_PRESTART_RETRIES.json',dict(UTC=utc(),tasks=approved,source_seal_unchanged=True,clock_sha256=sha(R/'CLOCK.json'),no_new_model_design=True))
pending=list(approved);running={};completed=[]
while pending or running:
    assert time.time()<clock['model_deadline_epoch']-60
    current=run(S+['systemctl --user list-units --state=running --no-legend --plain "logika-twelve-cpu2003-*"']).stdout
    slots=max(0,3-len([line for line in current.splitlines()if '.service'in line])-len(running))
    while pending and slots>0:
        task=pending.pop(0);unit='logika-twelve-prestart-retry-'+str(len(completed)+len(running))+'-20261029';seconds=min(1850,int(clock['model_deadline_epoch']-time.time()-30))
        cmd=['systemd-run','--user','--quiet','--unit='+unit,'--property=MemoryMax=10G','--property=CPUQuota=200%','--property=RuntimeMaxSec='+str(seconds),'--setenv=PYTHONPATH='+str(R/'ops/runtime_hooks'),'--property=StandardOutput=append:'+str(R/'logs'/f'{task}.retry2.log'),'--property=StandardError=append:'+str(R/'logs'/f'{task}.retry2.log'),'/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python','-u',str(R/'source/worker.py'),'--task',task,'--attempt','2']
        run(S+cmd);running[task]=unit;slots-=1
    for task,unit in list(running.items()):
        state=run(S+['systemctl','--user','show',unit,'-p','ActiveState','--value']).stdout.strip()
        if state in ['active','activating']:continue
        verify="""from pathlib import Path
import json,hashlib
r=Path(%r);d=r/'runs'/%r;second=d/'attempt2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=json.load(open(second/'RESULT.json'));assert v['status']=='complete' and v['attempt']==2;assert sha(second/'prediction.npz')==v['prediction_sha256'] and sha(second/'START.json')==v['start_sha256'];assert sha(d/'attempt1/FAILURE.json')==%r
archive=r/'failures/node2003_prestart'/%r;archive.mkdir(parents=True,exist_ok=True);(d/'attempt1').rename(archive/'attempt1');(d/'attempt1').symlink_to('attempt2',target_is_directory=True)
print(json.dumps({'result_sha256':sha(second/'RESULT.json'),'prediction_sha256':v['prediction_sha256']}))
"""%(str(R),task,approved[task]['original_failure_sha256'],task)
        verified=remote(verify)
        archive=R/'failures/node2003_prestart'/task;archive.mkdir(parents=True,exist_ok=True)
        run(['rsync','-az','-e','ssh -p2003',HOST+':'+str(archive)+'/',str(archive)+'/'])
        d=R/'runs'/task;d.mkdir(parents=True,exist_ok=True)
        run(['rsync','-az','-e','ssh -p2003',HOST+':'+str(d/'attempt2'),str(d)+'/'])
        assert sha(d/'attempt2/RESULT.json')==verified['result_sha256']and sha(d/'attempt2/prediction.npz')==verified['prediction_sha256']
        old=d/'attempt1'
        if old.exists() and not old.is_symlink():
            assert sha(old/'FAILURE.json')==approved[task]['original_failure_sha256']
            localarchive=R/'failures/canonical_prestart'/task;localarchive.mkdir(parents=True,exist_ok=True);old.rename(localarchive/'attempt1')
        if not old.exists():old.symlink_to('attempt2',target_is_directory=True)
        completed.append(dict(task=task,**verified));del running[task]
    dump(R/'NODE2003_REPAIR_STATE.json',dict(UTC=utc(),status='explicit_retry2_running',original_prestart_failures=12,retries_complete=completed,running=running,pending=pending,original_clock_preserved=True,source_unchanged=True))
    time.sleep(5)
run(S+['systemctl','--user','kill','--kill-whom=main','--signal=SIGCONT','logika-twelvehour-cpu2003-20261029.service'])
dump(R/'NODE2003_REPAIR_FINAL.json',dict(status='PASS',UTC=utc(),prestart_failures_preserved=12,successful_explicit_retry2=len(completed),tasks=completed,ordinary_queue_resumed=True,source_and_parameters_and_clock_unchanged=True,scientific_conclusion_from_prestart_failures=False))
