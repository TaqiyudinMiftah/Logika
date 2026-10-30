from pathlib import Path
import subprocess,concurrent.futures,shlex,json,time
R=Path(__file__).resolve().parent
routes=[('lab2003',['ssh','-p','2003'],'faris@gpu-labkc.ub.ac.id','/mnt/home/faris'),('host119',['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119'],'faris@100.89.211.119','/home/faris'),('host42',['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119'],'faris@100.98.169.42','/home/faris')]
def collect(route):
    name,cmd,host,home=route;dest=R/'private/incoming'/name;dest.mkdir(parents=True,exist_ok=True)
    for leaf in ['runs','logs','ownership']:
        target=dest/leaf;target.mkdir(exist_ok=True)
        q=subprocess.run(['rsync','-a','--exclude=*.tmp','-e',shlex.join(cmd),host+':'+home+'/logika-midnight-multiseed-20261030/'+leaf+'/',str(target)+'/'],capture_output=True)
        if q.returncode:raise RuntimeError(name+' '+leaf+' '+q.stderr.decode())
    q=subprocess.run(['/usr/bin/python3',str(R/'atomic_collect.py'),'--owner',name,'--source',str(dest)],capture_output=True,text=True)
    assert q.returncode==0,q.stderr
    return dict(owner=name,epoch=time.time(),complete_fits=int(q.stdout.strip()))
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as p:results=list(p.map(collect,routes))
report=dict(epoch=time.time(),owners=results,hash_verified_by_atomic_collector=True)
t=R/'COLLECTION_STATE.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(R/'COLLECTION_STATE.json');print(json.dumps(report),flush=True)
