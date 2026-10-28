"""Checkpoint-preserving replacement of one campaign-owned PC1/PC3 service."""
import argparse
from pathlib import Path
import shutil
import subprocess
import time
from metaexp.common import read,write,sha,code_hash
from node_executor import start

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True,choices=['pc1','pc3']);args=parser.parse_args()
    root=Path(__file__).resolve().parent;cfg=read(root/'campaign.json');revision=read(root/'scheduler_revision.json')
    if code_hash(root)!=cfg['code_hash'] or sha(root/'node_executor.py')!=revision['node_executor_sha256']:raise ValueError('Source identity differs')
    unit='metaexp-'+cfg['id']+'-'+args.host
    pid=int(subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','--value'],capture_output=True,text=True,check=True).stdout.strip() or 0)
    if pid:
        command=Path('/proc/'+str(pid)+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(root) not in command:raise ValueError('Service is outside this campaign')
        if 'node_executor.py' in command:print('Independent node executor already active');return
    folder=root/'revisions'/('local_executor_'+revision['node_executor_sha256'][:12]);folder.mkdir(parents=True,exist_ok=True)
    heartbeat=read(root/'heartbeat.json');write(folder/'before_heartbeat.json',heartbeat);checkpoints={}
    for tid in heartbeat['active']:
        run=root/'runs'/tid;checkpoints[tid]={str(p.relative_to(run)):sha(p) for p in run.glob('fold_*/*') if p.is_file() and (p.parent/'complete.json').exists()}
    write(folder/'checkpoint_hashes.json',checkpoints)
    subprocess.run(['systemctl','--user','stop',unit],check=True,timeout=40)
    resumed=[]
    for tid in heartbeat['active']:
        run=root/'runs'/tid;status=run/'status.json'
        if not status.exists():continue
        if (run/'complete.json').exists():write(status,{'state':'completed','finished_during_executor_replacement':True});continue
        write(run/('interruption_'+str(int(time.time()))+'.json'),{'status':read(status),'reason':'Independent local executor; task identity and completed folds preserved','resources':read(run/'resources.json') if (run/'resources.json').exists() else None})
        write(status,{'state':'running','resume_requested':True});resumed.append(tid)
    for name in ['daemon_identity.json','daemon_stopped.json']:
        if (root/name).exists():shutil.copy2(root/name,folder/name);(root/name).unlink()
    result=start(root,args.host);write(folder/'migration.json',{'time':time.time(),'resumed_jobs':resumed,'started':cfg['started'],'deadline':cfg['deadline'],'source_hash':cfg['code_hash'],'service':result})
    print({'host':args.host,'resumed_jobs':len(resumed),'deadline':cfg['deadline'],'service':result},flush=True)

if __name__=='__main__':main()
