"""Record real Kumo process resource usage and termination; no dummy allocation."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'

def dump(value,path):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def memory():
    p=Path('/sys/fs/cgroup')
    events=dict(l.split() for l in (p/'memory.events').read_text().splitlines())
    stats=dict(l.split() for l in (p/'memory.stat').read_text().splitlines())
    return dict(current_GiB=int((p/'memory.current').read_text())/2**30,limit_GiB=int((p/'memory.max').read_text())/2**30,inactive_file_GiB=int(stats.get('inactive_file',0))/2**30,events={k:int(v) for k,v in events.items()})

def gpu_free():
    try:
        r=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
        return float(r.stdout.strip().splitlines()[0])/1024
    except (subprocess.SubprocessError,ValueError,IndexError,OSError):return None

def wait_capacity(args,out):
    while True:
        snapshot=memory();unused=snapshot['limit_GiB']-snapshot['current_GiB'];reusable=min(snapshot['inactive_file_GiB'],.75)
        free_gpu=gpu_free();ram_ok=unused+reusable>=2.8;gpu_ok=free_gpu is not None and free_gpu>=24.
        if ram_ok and gpu_ok:return
        dump(dict(status='WAITING_FOR_RAM' if not ram_ok else 'WAITING_FOR_GPU',experiment=args.run,pid=os.getpid(),UTC=time.time(),memory=snapshot,GPU_free_GiB=free_gpu,required_available_plus_cache_GiB=2.8,required_GPU_free_GiB=24.,check_interval_seconds=10,note='Automatically launch real model work when both budgets fit; no training data or reservation buffers held while waiting'),out/'STATUS.json')
        time.sleep(10)

def run_once(args,worker,out):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True')
    with (V/'logs'/f'{args.run}_kumo_gpu.log').open('a') as log:
        p=subprocess.Popen(['/home/fariselhakim/adit/.venv-kumo-gpu-20261027/bin/python','-u',str(worker),'--run',args.run],stdout=log,stderr=subprocess.STDOUT,env=env)
        row=dict(pid=p.pid,supervisor_pid=os.getpid(),started=time.time(),memory_before=memory(),max_rss_GiB=0.,rss_limit_GiB=args.max_rss_gib)
        while p.poll() is None:
            try:
                fields={l.split(':',1)[0]:l.split(':',1)[1].strip() for l in Path(f'/proc/{p.pid}/status').read_text().splitlines() if ':' in l}
                rss=int(fields.get('VmRSS','0 kB').split()[0])/2**20
                row.update(current_rss_GiB=rss,max_rss_GiB=max(row['max_rss_GiB'],rss),last_sample=time.time(),memory_current=memory())
                if rss>args.max_rss_gib:
                    row['termination_reason']='Process RSS exceeded configured real-work memory budget';p.terminate()
                dump(row,out/'PROCESS_RESOURCE_STATUS.json')
            except FileNotFoundError:pass
            time.sleep(2)
        row.update(exit_code=p.returncode,ended=time.time(),memory_after=memory())
    if p.returncode and 'termination_reason' not in row:row['termination_reason']='Process exited nonzero; inspect log. cgroup OOM counter changes alone do not identify victim.'
    return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--max-rss-gib',type=float,default=5.5);parser.add_argument('--worker',default='kumo_gpu_experiment.py');args=parser.parse_args()
    worker=(V/'scripts'/args.worker).resolve();assert worker.parent==(V/'scripts').resolve() and worker.is_file()
    out=V/'runs'/args.run;out.mkdir(parents=True,exist_ok=True)
    attempts=out/'PROCESS_ATTEMPTS.json'
    history=json.loads(attempts.read_text()) if attempts.exists() else []
    for retry in range(3):
        wait_capacity(args,out)
        row=run_once(args,worker,out);row['resource_retry_index']=retry
        history.append(row);dump(history,attempts);dump(row,out/'PROCESS_RESOURCE_STATUS.json')
        if row['exit_code']==0:return
        if row['exit_code']==-9 and retry<2 and not (out/'RESULT.json').exists():
            dump(dict(status='WAITING_FOR_RAM_RETRY',experiment=args.run,pid=os.getpid(),UTC=time.time(),previous_exit_code=-9,retries_remaining=2-retry,note='Wait for capacity and resume immutable completed folds; no tight relaunch loop'),out/'STATUS.json')
            time.sleep(10);continue
        status=json.loads((out/'STATUS.json').read_text()) if (out/'STATUS.json').exists() else {}
        status.update(status='FAILED_PROCESS_EXIT',exit_code=row['exit_code'],resource_receipt=str(attempts),ended=row['ended']);dump(status,out/'STATUS.json');return

if __name__=='__main__':main()
