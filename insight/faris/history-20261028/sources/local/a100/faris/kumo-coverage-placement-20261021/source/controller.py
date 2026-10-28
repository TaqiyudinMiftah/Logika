"""Independent ordinary worker queue, with resource and absolute-deadline admission."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import subprocess,json,time,signal,shutil,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
GIB=1024**3
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def live(pid):
    try:return str(ROOT).encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
    except FileNotFoundError:return False
def main(host):
    import fcntl
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ROOT/'plan.json');assert read(ROOT/'SCIENTIFIC_GATE.json')['status']=='passed_for_bounded_development_study'
    path=ROOT/'controller-state.json'
    state=read(path)if path.exists()else dict(host=host,deadline=plan['original_deadline'],jobs={
        f'{f}_{v}':dict(fold=f,variant=v,status='pending')for f in plan['assigned_folds']for v in plan['conditions']})
    assert state['deadline']==plan['original_deadline']
    active={}
    # A restart never duplicates a live or uncertain previous worker.
    for jid,j in state['jobs'].items():
        if j['status']=='running':
            if live(j['pid']):active[jid]=None
            else:j.update(status='partial',reason='Previous worker no longer exists; explicit review needed before retry.')
    source=ROOT/'source/experiment.py';python=plan['python'];(ROOT/'logs').mkdir(exist_ok=True)
    while True:
        now=time.time()
        for jid in list(active):
            j=state['jobs'][jid];proc=active[jid]
            if not live(j['pid']):
                rc=proc.poll()if proc else None;d=ROOT/f'conditions/fold{j["fold"]}/{j["variant"]}'
                j.update(status='complete'if (d/'complete.json').exists() and rc in [None,0]else 'failed',returncode=rc,finished=now)
                del active[jid]
            elif now>=state['deadline'] or (ROOT/'STOP').exists() or now-j['started']>5400:
                # This PID was spawned with its own process group and its command
                # was just verified to belong to this exact campaign root.
                os.killpg(j['pid'],signal.SIGTERM);j['termination_requested']=True
        stop=(ROOT/'STOP').exists()or now>=state['deadline']-plan.get('verification_reserve_seconds',10800)
        if not stop:
            listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            used={state['jobs'][jid]['gpu']for jid in active}
            available=[int(a)for a,m,u in [line.split(',')for line in listing.splitlines()]if int(a)not in used and float(m)>=11000 and float(u)<10]
            for gpu in available:
                free=shutil.disk_usage(ROOT).free
                floor=2 if host=='pc3'else 25
                reservation=(len(active)+1)*plan.get('cache_reservation_GiB',7)+floor
                if free<reservation*GIB:break
                retained=sum(p.stat().st_size for p in ROOT.glob('conditions/*/*/fitted_context.pt'))
                if retained+(len(active)+1)*7*GIB>125*GIB:break
                candidates=[(k,j)for k,j in state['jobs'].items()if j['status']=='pending'and (ROOT/f'shared/fold{j["fold"]}/complete.json').exists()]
                if not candidates:break
                jid,j=candidates[0];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                log=(ROOT/'logs'/f'{jid}.log').open('ab')
                cmd=[python,str(source),'condition','--fold',str(j['fold']),'--variant',j['variant']]
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=ROOT)
                log.close();active[jid]=proc;j.update(status='running',pid=proc.pid,gpu=gpu,started=now,command=cmd)
        if stop:
            for j in state['jobs'].values():
                if j['status']=='pending':j.update(status='unstarted',reason='Absolute deadline or finalization reserve admission limit')
        state.update(updated=now,status='running'if active or any(j['status']=='pending'for j in state['jobs'].values())else 'terminal')
        write(path,state)
        if state['status']=='terminal':break
        time.sleep(10)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',choices=['pc2','pc3'],required=True)
    main(parser.parse_args().host)
