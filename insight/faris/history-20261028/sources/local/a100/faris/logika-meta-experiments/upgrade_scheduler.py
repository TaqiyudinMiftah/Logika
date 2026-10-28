"""Replace only the campaign coordinator, preserving its original clock and fits."""
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from metaexp.common import read,write,sha,code_hash
from metaexp.coordinator import Transport

def main():
    root=Path(__file__).resolve().parent;cfg=read(root/'campaign.json');before=(root/'campaign.json').read_bytes()
    if code_hash(root)!=cfg['code_hash']:raise ValueError('Training source changed')
    if time.time()>=cfg['deadline']:raise ValueError('Original campaign deadline expired')
    revision=read(root/'scheduler_revision.json');digest=sha(root/'scheduler_parallel.py')
    if digest!=revision['script_sha256'] or sha(root/'rpc_parallel.py')!=revision['rpc_sha256']:raise ValueError('Untested scheduler source')
    archive=root/'revisions'/('parallel_'+digest[:12]);archive.mkdir(parents=True,exist_ok=True)
    unit='metaexp-'+cfg['id']+'-coordinator'
    pid=int(subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','--value'],capture_output=True,text=True,check=True).stdout.strip() or 0)
    if pid:
        command=Path('/proc/'+str(pid)+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(root) not in command:raise ValueError('Refusing to stop a process outside this campaign')
        status=read(root/'campaign_status.json')
        if 'scheduler_parallel.py' in command and status.get('scheduler',{}).get('revision')==digest:print(json.dumps({'already_upgraded':True,'pid':pid,'deadline':cfg['deadline']}));return
    for name in ['campaign.json','launch.json','campaign_status.json','daemon_identity.json','heartbeat.json']:
        if (root/name).exists():shutil.copy2(root/name,archive/('before_'+name))
    with sqlite3.connect(root/'ledger.sqlite') as source,sqlite3.connect(archive/'ledger_before.sqlite') as dest:source.backup(dest)
    hb=read(root/'heartbeat.json');active=list(hb.get('active',[]));checkpoints={}
    for tid in active:
        folder=root/'runs'/tid
        checkpoints[tid]={str(p.relative_to(folder)):sha(p) for p in folder.glob('fold_*/*') if p.is_file() and (p.parent/'complete.json').exists()}
    write(archive/'checkpoint_hashes.json',checkpoints)
    # Keep PC1's login alive across the brief coordinator replacement. This is
    # a setup-only SSH session, and is closed after the new resident hold starts.
    log=(archive/'transition_hold.log').open('ab');tx=Transport(root,cfg)
    hold=subprocess.Popen(tx.command('pc1'),stdin=subprocess.PIPE,stdout=log,stderr=log)
    hold.stdin.write(json.dumps({'action':'hold'}).encode());hold.stdin.close()
    try:
        subprocess.run(['systemctl','--user','stop',unit],check=True,timeout=40)
        resumed=[]
        for tid in active:
            folder=root/'runs'/tid;p=folder/'status.json'
            if not p.exists():continue
            old=read(p)
            if (folder/'complete.json').exists():write(p,{'state':'completed','finished_during_scheduler_replacement':True});continue
            write(folder/('interruption_'+str(int(time.time()))+'.json'),{'status':old,'reason':'Scheduler-only replacement; completed folds preserved','resources':read(folder/'resources.json') if (folder/'resources.json').exists() else None})
            write(p,{'state':'running','resume_requested':True,'reason':'Scheduler-only replacement, same task identity and checkpoints'});resumed.append(tid)
        for name in ['daemon_identity.json','daemon_stopped.json']:
            if (root/name).exists():shutil.copy2(root/name,archive/('stopped_'+name));(root/name).unlink()
        with sqlite3.connect(root/'ledger.sqlite') as db:
            for tid in resumed:db.execute("UPDATE tasks SET state='assigned',verified=0,finished=NULL WHERE id=? AND verified=0 AND state IN ('assigned','running','partial')",(tid,))
        original=read(archive/'before_launch.json');args=original['command']
        index=args.index(str(root/'.venv/bin/python'));args=args[:index]+[str(root/'.venv/bin/python'),'-u',str(root/'scheduler_parallel.py'),'--root',str(root)]
        args=[('--property=RuntimeMaxSec='+str(int(cfg['deadline']-time.time()+5))) if value.startswith('--property=RuntimeMaxSec=') else value for value in args]
        subprocess.run(['systemctl','--user','reset-failed',unit],capture_output=True)
        result=subprocess.run(args,capture_output=True,text=True,timeout=20)
        if result.returncode:raise RuntimeError(result.stderr)
        write(root/'launch.json',{'unit':unit,'started':cfg['started'],'deadline':cfg['deadline'],'command':args,'scheduler_revision':digest,'previous_launch':str(archive/'before_launch.json')})
        started=time.time();ready=False
        while time.time()-started<50:
            status=read(root/'campaign_status.json')
            if status.get('scheduler',{}).get('revision')==digest:ready=True;break
            time.sleep(1)
        if not ready:raise RuntimeError('Replacement has not yet published a fresh heartbeat; inspect the owned service')
        if (root/'campaign.json').read_bytes()!=before:raise ValueError('Original clock/configuration unexpectedly changed')
        for tid,files in checkpoints.items():
            for file,want in files.items():
                # A fold already complete before replacement must never be refit.
                if (root/'runs'/tid/Path(file).parent/'complete.json').exists() and sha(root/'runs'/tid/file)!=want:raise ValueError('Completed checkpoint changed')
        write(archive/'migration.json',{'time':time.time(),'started':cfg['started'],'deadline':cfg['deadline'],'campaign_sha256':sha(root/'campaign.json'),'training_code_hash':code_hash(root),'scheduler_revision':digest,'resumed_partial_jobs':resumed,'previous_main_pid':pid,'new_status':status})
        print(json.dumps({'upgraded':True,'same_original_deadline':cfg['deadline'],'resumed_jobs':resumed,'counts':status['counts'],'scheduler':status['scheduler']}),flush=True)
    finally:
        if hold.poll() is None:hold.terminate()
        log.close()

if __name__=='__main__':main()
