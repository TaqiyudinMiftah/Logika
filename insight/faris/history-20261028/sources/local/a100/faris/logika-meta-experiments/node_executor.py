"""Independent node executor; delegates fitting to the unchanged original worker."""
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import psutil
from metaexp.common import read,sha,code_hash,event

def queue_read(path):
    value=read(path)
    if Path(path).parent.name=='inbox' and value.get('execution_timeout_seconds'):
        # The persisted task retains its phase deadline. Only the daemon's
        # in-memory admission record receives the per-attempt wall deadline.
        value=dict(value);value['until']=min(value['until'],time.time()+float(value['execution_timeout_seconds']))
    return value

def start(root,host):
    root=Path(root).resolve();cfg=read(root/'campaign.json');machine=cfg['machines'][host]
    with (root/'node_start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        record=root/'daemon_identity.json'
        if record.exists():
            old=read(record)
            try:
                process=psutil.Process(old['pid'])
                if abs(process.create_time()-old['create_time'])<.01:return {'already_running':True,'pid':process.pid}
            except psutil.NoSuchProcess:pass
        if time.time()>=cfg['deadline']:raise ValueError('Original deadline expired')
        if host=='pc2':
            with (root/'daemon.log').open('ab') as log:
                p=subprocess.Popen([str(root/'.venv/bin/python'),'-u',str(root/'node_executor.py'),'--root',str(root),'--host',host],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
            return {'pid':p.pid}
        unit='metaexp-'+cfg['id']+'-'+host
        condition=str(root/'.venv/bin/python')+' -c '+shlex.quote('import time,sys;sys.exit(0 if time.time()<'+repr(cfg['deadline'])+' else 1)')
        subprocess.run(['systemctl','--user','reset-failed',unit],capture_output=True)
        args=['systemd-run','--user','--unit='+unit,'--property=CPUQuota='+str(machine['cpu_quota'])+'%',
              '--property=MemoryMax='+str(machine['memory_gib'])+'G','--property=MemorySwapMax=0','--property=OOMScoreAdjust=1000',
              '--property=Restart=on-failure','--property=RestartSec=5','--property=KillMode=control-group','--property=WorkingDirectory='+str(root),
              '--property=ExecCondition='+condition,'--property=RuntimeMaxSec='+str(int(cfg['deadline']-time.time()+5)),
              str(root/'.venv/bin/python'),'-u',str(root/'node_executor.py'),'--root',str(root),'--host',host]
        result=subprocess.run(args,capture_output=True,text=True,timeout=20)
        if result.returncode:raise RuntimeError(result.stderr)
        return {'unit':unit}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--host',required=True);args=parser.parse_args()
    root=args.root.resolve();cfg=read(root/'campaign.json');revision=read(root/'scheduler_revision.json')
    if sha(Path(__file__))!=revision['node_executor_sha256'] or code_hash(root)!=cfg['code_hash']:raise ValueError('Executor or frozen training source differs')
    from metaexp import runtime
    runtime.read=queue_read
    event(root/'node_execution.jsonl',event='local_executor_started',host=args.host,executor_sha256=sha(Path(__file__)),deadline=cfg['deadline'],training_code_hash=cfg['code_hash'])
    runtime.daemon(root,args.host)

if __name__=='__main__':main()
