"""Two independent GPU workers supervised by a finite, non-AI controller."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import fcntl,hashlib,importlib.metadata,json,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import baseline as b

def write_state(s):b.dump(s,ROOT/'state.json')
def event(v):
    v={'time':time.time(),**v};print(json.dumps(v),flush=True)
    with (ROOT/'controller-events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def free_gpus():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return {int(i):float(m) for i,m in [line.split(',') for line in p.stdout.splitlines()]}
def run():
    lock=(ROOT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'plan.json').read_text());seal=json.loads((ROOT/'seal.json').read_text())
    for name,h in seal['files'].items():assert b.sha(ROOT/name)==h,name
    for name,h in seal['inputs'].items():assert b.sha(name)==h,name
    assert not (ROOT/'state.json').exists(),'Preserve the original attempt and clock'
    env={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','torch','structured-data-models','pyarrow','huggingface_hub','safetensors','threadpoolctl']}
    for p,v in plan['versions'].items():assert env[p]==v,(p,env[p],v)
    b.dump(env,ROOT/'environment.json')
    start=time.time();s={'start':start,'deadline':start+plan['max_seconds'],'status':'training','completed_folds':[],'active':{}}
    write_state(s);pending=list(range(5));active={}
    try:
        import pipeline
        pipeline.preflight(ROOT);event({'stage':'nested_boundaries_and_baseline_verified'})
        while pending or active:
            assert time.time()<s['deadline'] and not (ROOT/'STOP').exists(),'Deadline or stop request'
            if pending:
                free=free_gpus()
                for gpu in [0,1]:
                    if gpu in active or not pending or free.get(gpu,0)<plan['min_free_vram_mib']:continue
                    f=pending.pop(0);log=(ROOT/f'fold{f}.log').open('ab');e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu)
                    p=subprocess.Popen([sys.executable,str(ROOT/'source/pipeline.py'),'fold','--root',str(ROOT),'--fold',str(f)],
                        stdout=log,stderr=log,env=e,start_new_session=True)
                    active[gpu]=(p,log,f);s['active'][str(gpu)]={'pid':p.pid,'fold':f}
                    event({'started_outer_fold':f,'gpu':gpu,'pid':p.pid})
            for gpu,(p,log,f) in list(active.items()):
                if p.poll() is not None:
                    log.close();del active[gpu];s['active'].pop(str(gpu),None)
                    assert p.returncode==0 and (ROOT/f'folds/fold{f}/complete.json').exists(),f'Fold {f} failed; see fold{f}.log'
                    s['completed_folds'].append(f);event({'completed_outer_fold':f})
            write_state(s);time.sleep(5)
        s['status']='evaluating';write_state(s)
        with (ROOT/'report.log').open('ab') as log:
            p=subprocess.Popen([sys.executable,str(ROOT/'source/pipeline.py'),'finish','--root',str(ROOT)],
                stdout=log,stderr=log,start_new_session=True)
            try:
                while p.poll() is None:
                    assert time.time()<s['deadline'] and not (ROOT/'STOP').exists()
                    time.sleep(5)
                assert p.returncode==0,'Final assessment failed; see report.log'
            finally:stop(p)
        s['status']='complete';s['result']=json.loads((ROOT/'result.json').read_text())
    except BaseException:
        s['status']='failed';s['error']=traceback.format_exc()
        for p,log,f in active.values():stop(p);log.close()
        (ROOT/'REPORT.md').write_text('# Incomplete training\n\n'+s['error']+'\nAll produced checkpoints and outputs remain available. No pipeline was promoted.\n')
        raise
    finally:
        s['finished']=time.time();s['active']={};write_state(s);event({'DONE':True,'status':s['status']})

if __name__=='__main__':run()
