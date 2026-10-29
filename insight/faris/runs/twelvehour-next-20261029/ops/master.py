"""Canonical laptop-independent launch, collection, freeze and terminal audit."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shlex,shutil,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import load,dump,sha,utc,check_seal,tasks,catalog
sys.path.insert(0,str(R/'ops'));from atomic_collect import publish
PY='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
SSH3=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2003','faris@gpu-labkc.ub.ac.id']
SSH119=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119','faris@100.89.211.119']
OWNER_FILES=['CONFIG.json','SOURCE_SEAL.json','VALIDATION_CONTRACT.json','INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','EXECUTION_PLAN.json','CLOCK.json','GATE.json','TRAINING_BRIEF.md','DEADLINE.md','TRAINING_STATE.json','AGENTS.md','PREPUBLICATION.json','EXTRA_DESIGNS.jsonl','RESOURCE_PREPUBLICATION.json','COLAB_RESOURCE_ADDENDUM.json']
def call(args,timeout=180):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
def module(name,extra=()):
    remaining=load('CLOCK.json')['deadline_epoch']-time.time()-20
    if remaining<20:raise TimeoutError('Original verification ceiling reached')
    with(R/'logs'/f'{name}.log').open('ab')as stream:
        subprocess.run([PY,str(R/'source'/f'{name}.py'),*extra],check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
def start_unit(unit,python,script,args,root,ssh=None,memory='4G',quota='200%',end=None):
    seconds=int((end or load('CLOCK.json')['deadline_epoch'])-time.time()-10);assert seconds>0
    command=['systemd-run','--user','--quiet','--unit='+unit,'--property=MemoryMax='+memory,'--property=CPUQuota='+quota,'--property=KillMode=control-group','--property=RuntimeMaxSec='+str(seconds),'--property=StandardOutput=append:'+str(root/'service.log'),'--property=StandardError=append:'+str(root/'service.log'),python,'-u',str(script),*args]
    call([*(ssh or []),*command],60)
def create_owner(owner,remote=False):
    dest=R/'private/backends'/owner
    if remote:
        call(SSH3+['mkdir','-p',str(dest)])
        for name in [*OWNER_FILES,'source','ops','evidence']:
            call(['rsync','-az','-e','ssh -o BatchMode=yes -p2003',str(R/name),'faris@gpu-labkc.ub.ac.id:'+str(dest)+'/'])
        code="from pathlib import Path;r=Path("+repr(str(dest))+");p=Path('/mnt/home/faris/logika-twelvehour-20261029/packets');assert p.is_dir();(r/'packets').symlink_to(p,target_is_directory=True)if not(r/'packets').exists()else None;[(r/n).mkdir(exist_ok=True)for n in ['logs','private','runs']]"
        call(SSH3+['python3 -c '+shlex.quote(code)])
    else:
        dest.mkdir(parents=True,exist_ok=True)
        for name in OWNER_FILES:shutil.copy2(R/name,dest/name)
        for name in ['source','ops','evidence']:shutil.copytree(R/name,dest/name,dirs_exist_ok=True)
        if not(dest/'packets').exists():(dest/'packets').symlink_to(R/'packets',target_is_directory=True)
        for name in ['logs','private','runs']:(dest/name).mkdir(exist_ok=True)
    python='/mnt/home/faris/logika-submit-best-sub25k-20261029/env_gpu/bin/python'if owner=='gpu2003'else PY
    slots=load('EXECUTION_PLAN.json')['owners'][owner]['slots']
    start_unit('logika-next-'+owner+'-20261029',python,dest/'ops/owner_queue.py',['--owner',owner],dest,SSH3 if remote else None,memory=('24G'if owner=='gpu2003'else'28G'),quota=str(slots*200)+'%',end=load('CLOCK.json')['model_deadline_epoch'])
    return dest
def collect(owner,remote=False):
    src=str(R/'private/backends'/owner/'runs')
    if remote:src='faris@gpu-labkc.ub.ac.id:'+src;os.environ['LOGIKA_COLLECT_RSH']='ssh -o BatchMode=yes -p2003'
    try:result=publish(src,owner)
    finally:os.environ.pop('LOGIKA_COLLECT_RSH',None)
    dst=R/'private/owner_metadata'/owner;dst.mkdir(parents=True,exist_ok=True)
    if remote:
        call(['rsync','-az','-e','ssh -o BatchMode=yes -p2003','--include=/*.json','--exclude=*','faris@gpu-labkc.ub.ac.id:'+str(R/'private/backends'/owner)+'/',str(dst)+'/'],60)
    else:
        for p in (R/'private/backends'/owner).glob('*.json'):shutil.copy2(p,dst/p.name)
    if(dst/'BACKEND_FINAL.json').exists():dump(dict(UTC=utc(),owner=owner,collected=True,receipt=result,terminal_metadata_sha256=sha(dst/'BACKEND_FINAL.json')),R/f'BACKEND_COLLECTION_{owner}_COMPLETE.json')
    return result
def prepare_host119():
    dest=Path('/home/faris/logika-twelvehour-next-20261029');call(SSH119+['mkdir','-p',str(dest)])
    transport='ssh -o BatchMode=yes -i /mnt/home/faris/.ssh/id_ed25519_logika_backup119'
    for name in [*OWNER_FILES,'source','ops','evidence','private/DRIVE_FOLDER.json']:
        if(R/name).exists():
            target=dest/('private/'if name.startswith('private/')else'')
            call(SSH119+['mkdir','-p',str(target)])
            call(['rsync','-az','-e',transport,str(R/name),'faris@100.89.211.119:'+str(target)+'/'])
    call(SSH119+['python3 -c '+shlex.quote("from pathlib import Path;r=Path('/home/faris/logika-twelvehour-next-20261029');[(r/n).mkdir(exist_ok=True)for n in ['logs','runs','private']]")])
    start_unit('logika-next-host119-20261029','/usr/bin/python3',dest/'ops/host_manager.py',[],dest,SSH119,memory='6G',quota='200%')
def launch():
    (R/'private').mkdir(exist_ok=True);(R/'logs').mkdir(exist_ok=True);(R/'runs').mkdir(exist_ok=True)
    lock=(R/'master.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not(R/'CLOCK.json').exists(),'Fresh launch only. Reconcile actual units before any execution repair; never reset clock.'
    receipt=load('PREPUBLICATION.json');assert len(receipt['remote_commit'])==40 and receipt['config_sha256']==sha(R/'CONFIG.json')and receipt['source_seal_sha256']==sha(R/'SOURCE_SEAL.json')
    call(['/usr/bin/python3',str(R/'ops/init_competition.py'),str(R),'logika-next-20261029','pooled absolute binary KS','higher','12','--train',str(R/'supplied_data/train_applications.csv'),'--test',str(R/'supplied_data/test_applications.csv'),'--compute','11explicit Colab profiles; lab2001 CPU4slots, lab2003 CPU4slots and2idleRTX3060, host119 CPU6slots and persistent control;2threads perfit','--rules',str(R/'evidence/rules.md')])
    st=load('TRAINING_STATE.json');start=datetime.datetime.fromisoformat(st['start_utc']).timestamp();cfg=load('CONFIG.json')
    dump(dict(started_epoch=start,started_UTC=st['start_utc'],admission_end_epoch=start+10.5*3600,model_deadline_epoch=start+11.25*3600,deadline_epoch=start+12*3600,hours=12,config_sha256=sha(R/'CONFIG.json'),contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),execution_plan_sha256=sha(R/'EXECUTION_PLAN.json'),skill_deadline_sha256=sha(R/'DEADLINE.md')),R/'CLOCK.json');os.chmod(R/'CLOCK.json',0o444)
    clock=check_seal();module('gate');assert load('GATE.json')['status']=='PASS'
    dump(dict(remote_commit=receipt['remote_commit'],extra_designs_sha256=sha(R/'EXTRA_DESIGNS.jsonl'),source_seal_sha256=sha(R/'SOURCE_SEAL.json')),R/'RESOURCE_PREPUBLICATION.json')
    dump(dict(assignments=load('EXECUTION_PLAN.json')['owners'],all11profiles=True,no_quota_churn=True),R/'COLAB_RESOURCE_ADDENDUM.json')
    for owner,remote in [('cpu2001',False),('cpu2003',True),('gpu2003',True)]:create_owner(owner,remote)
    prepare_host119();last_eval=0;last_health=0;errors={}
    allowners=list(load('EXECUTION_PLAN.json')['owners'])
    while time.time()<clock['model_deadline_epoch']-90:
        for owner,remote in [('cpu2001',False),('cpu2003',True),('gpu2003',True)]:
            try:collect(owner,remote);errors.pop(owner,None)
            except Exception as e:errors[owner]=type(e).__name__
        module('compose')
        if time.time()-last_eval>300:module('evaluate');last_eval=time.time()
        complete=len(list((R/'runs').glob('*/attempt1/RESULT.json')));failed=len(list((R/'runs').glob('*/attempt1/FAILURE.json')));collected=[o for o in allowners if(R/f'BACKEND_COLLECTION_{"account"+o if o.isdigit()else o}_COMPLETE.json').exists()]
        state=dict(stage='running',UTC=utc(),completed_fits=complete,failed_attempts=failed,planned_fits=len(tasks()),registered_designs=len(catalog()),terminal_collected_owners=collected,owners=allowners,collection_errors=errors,clock_sha256=sha(R/'CLOCK.json'),deadline_epoch=clock['deadline_epoch'],laptop_required=False,automatic_refill=True,no_new_features=True,no_submission=True)
        dump(state,R/'LIVE_STATE.json')
        if time.time()-last_health>=1800:
            dump(dict(state,health_check='30minute resource/queue reconciliation; idle healthy slots already admit next predeclared completecohort immediately',next_epoch=time.time()+1800),R/'HEALTH_30MIN.json');last_health=time.time()
        if len(collected)==len(allowners):break
        time.sleep(20)
    # Stop only this campaign's units at the original model cutoff, then collect.
    for owner,ssh in [('cpu2001',None),('cpu2003',SSH3),('gpu2003',SSH3)]:
        subprocess.run([*(ssh or[]),'systemctl','--user','stop','logika-next-'+owner+'-20261029'],capture_output=True)
        try:collect(owner,ssh is not None)
        except Exception:pass
    # Host controller independently collects/releases its VMs before this barrier.
    while time.time()<clock['deadline_epoch']-1800:
        missing=[o for o in allowners if not(R/f'BACKEND_COLLECTION_{"account"+o if o.isdigit()else o}_COMPLETE.json').exists()]
        if not missing:break
        dump(dict(UTC=utc(),missing=missing,clock_sha256=sha(R/'CLOCK.json')),R/'TERMINAL_COLLECTION_WAIT.json');time.sleep(15)
    module('compose');module('evaluate',['--final']);module('audit')
    result=dict(stage='complete',UTC=utc(),completed_fits=len(list((R/'runs').glob('*/attempt1/RESULT.json'))),failed_attempts=len(list((R/'runs').glob('*/attempt1/FAILURE.json'))),planned_fits=len(tasks()),artifact_audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json'),final_evaluation_sha256=sha(R/'FINAL_EVALUATION.json'),clock_sha256=sha(R/'CLOCK.json'),no_refit=True,no_submission=True,laptop_required=False,source_seal_sha256=sha(R/'SOURCE_SEAL.json'))
    dump(result,R/'COMPLETION.json');dump(result,R/'LIVE_STATE.json')
if __name__=='__main__':
    try:launch()
    except BaseException:
        p=R/'private/MASTER_FAILURE.log';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc());dump(dict(status='execution_review_required',UTC=utc(),trace_sha256=sha(p),clock_preserved=True),R/'MASTER_FAILURE.json');raise
