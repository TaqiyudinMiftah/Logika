"""Remote-only health checks, redundant evidence backup and publication outbox.

No modelling, clock changes, restarts, credentials or leaderboard access.
"""
from pathlib import Path
import datetime, fcntl, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, time, traceback

R = Path(__file__).resolve().parents[1]
DEST = '/mnt/home/faris/logika-twelvehour-20261029-backup-2001'
HOST = 'faris@gpu-labkc.ub.ac.id'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-p', '2003', HOST]
CLOCK_SHA = None  # bound once from immutable CLOCK at daemon start
P = R / 'private/laptop-independent'

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def run(args, timeout=180): return subprocess.run(args,check=True,text=True,capture_output=True,timeout=timeout)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def backup():
    run(SSH+['mkdir','-p',DEST])
    args=['rsync','-az','-e','ssh -o BatchMode=yes -o ConnectTimeout=10 -p2003',
          '--exclude=private/***','--exclude=publication_ready/***','--exclude=__pycache__/***',
          '--include=*/','--include=*.json','--include=*.jsonl','--include=*.py',
          '--include=*.npz','--include=*.npy','--include=*.md','--include=*.log','--exclude=*',str(R)+'/',HOST+':'+DEST+'/']
    preview=run(args[:-2]+['--dry-run','--stats']+args[-2:]);match=re.search(r'Total transferred file size:\s*([\d,]+)',preview.stdout);assert match
    need=int(match.group(1).replace(',',''))
    free=int(run(SSH+['python3 -c '+repr('import shutil;print(shutil.disk_usage("/mnt/home/faris").free)')]).stdout.strip())
    assert free-need>1024**3,'Preserve1GiB backup-node reserve'
    run(args)
    code='''from pathlib import Path
import json,hashlib
r=Path(%r)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(r/'CLOCK.json')==%r
for name,value in json.loads((r/'SOURCE_SEAL.json').read_text()).items():assert sha(r/name)==value
checked=0
for p in (r/'runs').glob('*/attempt1/RESULT.json'):
 v=json.loads(p.read_text());assert sha(p.parent/'prediction.npz')==v['prediction_sha256'];assert sha(p.parent/'START.json')==v['start_sha256'];assert v['weights_saved'] is False;checked+=1
manifest={str(p.relative_to(r)):{'sha256':sha(p),'bytes':p.stat().st_size}for p in r.rglob('*')if p.is_file() and p.name!='BACKUP_MANIFEST.json'}
(r/'BACKUP_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True)+'\\n')
print(json.dumps({'status':'HASH_VERIFIED','completed_fit_records':checked,'files':len(manifest),'manifest_sha256':sha(r/'BACKUP_MANIFEST.json'),'clock_sha256':sha(r/'CLOCK.json')}))
'''%(DEST,CLOCK_SHA)
    import shlex
    result=json.loads(run(SSH+['python3 -c '+shlex.quote(code)]).stdout)
    dump(P/'BACKUP_RECEIPT.json',dict(result,UTC=utc(),host='node2003',path=DEST,case_predictions_private=True))
    return result

def export():
    pub=module('logika_pub',R/'ops/publish.py')
    helper=module('logika_history',R/'ops/publish_experiment_history.py')
    out=P/'PUBLICATION_OUTBOX';out.mkdir(exist_ok=True);files={}
    for p in sorted(R.rglob('*')):
        if not p.is_file():continue
        relative=p.relative_to(R)
        operational=relative.parts[0] in ['source','ops'] and p.suffix=='.py'
        if not operational and not pub.allowed_relative(relative):continue
        data=pub.safe_payload(p,helper.public_content)
        if data is None:continue
        q=out/relative;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
        files[str(relative)]=dict(sha256=sha(q),bytes=q.stat().st_size)
    scanner=R/'ops/gitleaks';report=P/'OUTBOX_SECRET_SCAN.json'
    run([str(scanner),'dir',str(out),'--config',str(R/'ops/gitleaks.toml'),'--ignore-gitleaks-allow','--redact=100','--no-banner','--no-color','--report-format','json','--report-path',str(report)])
    assert json.loads(report.read_text())==[]
    dump(P/'OUTBOX_MANIFEST.json',files)
    receipt=dict(status='SANITIZED_READY_FOR_GITHUB_SYNC',UTC=utc(),files=len(files),manifest_sha256=sha(P/'OUTBOX_MANIFEST.json'),secret_findings=0,github_sync='Resumes from laptop; no Git credentials copied to lab')
    dump(P/'OUTBOX_RECEIPT.json',receipt);return receipt

def snapshot():
    assert sha(R/'CLOCK.json')==CLOCK_SHA
    state=json.loads((R/'LIVE_STATE.json').read_text())
    units={}
    for name in ['logika-twelvehour-20261029.service']:
        units['node2001']=run(['systemctl','--user','show',name,'-p','ActiveState','-p','SubState','-p','MainPID']).stdout
    units['node2003']=run(SSH+['systemctl','--user','show','logika-twelvehour-cpu2003-20261029.service','-p','ActiveState','-p','SubState','-p','MainPID']).stdout
    b=backup();e=export()
    result=dict(status='PASS',UTC=utc(),training_requires_laptop=False,health_monitor_requires_laptop=False,backup_requires_laptop=False,github_push_requires_laptop=True,github_records_queued_remotely=True,clock_sha256=CLOCK_SHA,live_state=state,services=units,backup=b,outbox=e,scientific_files_unchanged=True)
    dump(R/'LAPTOP_INDEPENDENCE.json',result)
    with (P/'HEALTH_HISTORY.jsonl').open('a')as f:f.write(json.dumps(result)+'\n')
    return result

def main():
    global CLOCK_SHA
    CLOCK_SHA=sha(R/'CLOCK.json')
    P.mkdir(parents=True,exist_ok=True);lock=(P/'monitor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    clock=json.loads((R/'CLOCK.json').read_text());next_check=0
    while time.time()<clock['deadline_epoch']-15:
        terminal=(R/'COMPLETION.json').exists() and (R/'ARTIFACT_CONSISTENCY_AUDIT.json').exists()
        if time.time()>=next_check or terminal:
            try:
                result=snapshot();print(json.dumps({'UTC':utc(),'status':result['status'],'completed':result['backup']['completed_fit_records']}),flush=True)
                if terminal:
                    assert json.loads((R/'ARTIFACT_CONSISTENCY_AUDIT.json').read_text())['status']=='PASS'
                    dump(P/'FINAL_REMOTE_RECEIPT.json',dict(result,completion_sha256=sha(R/'COMPLETION.json'),audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json')));return
                next_check=time.time()+1800
            except Exception:
                (P/'FAILURE_PRIVATE.log').write_text(traceback.format_exc());dump(P/'MONITOR_FAILURE.json',dict(UTC=utc(),private_trace_sha256=sha(P/'FAILURE_PRIVATE.log'),training_uninterrupted=True));next_check=time.time()+60
        time.sleep(15)
    snapshot()

if __name__=='__main__':main()
