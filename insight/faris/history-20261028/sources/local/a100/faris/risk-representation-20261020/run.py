"""Finite two-GPU controller with an absolute deadline and retained artifacts."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import fcntl,json,signal,subprocess,sys,time,traceback,shutil
import importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import baseline as b

def save(s):b.dump(s,ROOT/'state.json')
def event(stage,**values):print(json.dumps(dict(time=time.time(),stage=stage,**values)),flush=True)
def stop(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def available():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True)
    return {int(a):float(b) for a,b in [line.split(',') for line in result.stdout.splitlines()]}
def permitted(state):
    assert time.time()<state['deadline'] and not (ROOT/'STOP').exists(),'Deadline/stop request'
    assert shutil.disk_usage(ROOT).free>16*1024**3,'Filesystem headroom is low'
    # Data-dependent admission is bounded to this campaign only.
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<120*1024**3,'Campaign storage ceiling'
def single(state,action,gpu=None):
    env=os.environ.copy()
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=(ROOT/(action+'.log')).open('ab')
    child=subprocess.Popen([sys.executable,str(ROOT/'source/experiment.py'),action,'--root',str(ROOT)],
        stdout=log,stderr=log,env=env,start_new_session=True)
    state['active_phase_pid']=child.pid;save(state)
    try:
        while child.poll() is None:permitted(state);time.sleep(3)
        assert child.returncode==0,f'{action} failed; see {action}.log'
    finally:
        stop(child);log.close();state.pop('active_phase_pid',None);save(state)
def run():
    lock=(ROOT/'controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'plan.json').read_text());seal=json.loads((ROOT/'seal.json').read_text())
    for name,digest in seal['files'].items():assert b.sha(ROOT/name)==digest,name
    for name,digest in seal['inputs'].items():assert b.sha(name)==digest,name
    versions={name:importlib.metadata.version(name) for name in plan['versions']}
    assert versions==plan['versions'];b.dump(versions,ROOT/'environment.json')
    if (ROOT/'state.json').exists():
        state=json.loads((ROOT/'state.json').read_text())
        assert state['status']!='complete','Already complete; preserve results'
        old_pids=[v['pid'] for v in state.get('active',{}).values()]
        if state.get('active_phase_pid'):old_pids.append(state['active_phase_pid'])
        for pid in old_pids:
            cmdline=Path(f'/proc/{pid}/cmdline')
            if cmdline.exists():
                assert str(ROOT).encode() not in cmdline.read_bytes(), 'Prior owned worker is still alive; do not duplicate it'
        event('resume',original_deadline=state['deadline'])
    else:
        now=time.time();state=dict(start=now,deadline=now+plan['max_seconds'],completed_folds=[])
    state.update(status='preflight',active={});save(state);active={}
    try:
        permitted(state)
        if not (ROOT/'preflight.json').exists():
            while available().get(0,0)<11000:permitted(state);time.sleep(10)
            single(state,'preflight',0)
        assert json.loads((ROOT/'preflight.json').read_text())['status']=='passed'
        state['status']='training';save(state)
        pending=[]
        for f in range(5):
            done=ROOT/f'folds/fold{f}/complete.json'
            if done.exists():
                meta=json.loads(done.read_text())
                for name,digest in meta['files'].items():assert b.sha(done.parent/name)==digest
                if f not in state['completed_folds']:state['completed_folds'].append(f)
            else:pending.append(f)
        while pending or active:
            permitted(state);free=available() if pending else {}
            for gpu in [0,1]:
                if gpu in active or not pending or free.get(gpu,0)<11000:continue
                f=pending.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                log=(ROOT/f'fold{f}.log').open('ab')
                child=subprocess.Popen([sys.executable,str(ROOT/'source/experiment.py'),'fold','--root',str(ROOT),'--fold',str(f)],
                    stdout=log,stderr=log,env=env,start_new_session=True)
                active[gpu]=(child,log,f);state['active'][str(gpu)]=dict(pid=child.pid,fold=f)
                event('outer_started',fold=f,gpu=gpu,pid=child.pid)
            for gpu,(child,log,f) in list(active.items()):
                if child.poll() is None:continue
                log.close();del active[gpu];state['active'].pop(str(gpu),None)
                assert child.returncode==0 and (ROOT/f'folds/fold{f}/complete.json').exists(),f'Fold {f} failed; see fold{f}.log'
                state['completed_folds'].append(f);event('outer_complete',fold=f)
            save(state);time.sleep(5)
        state['status']='reporting';save(state);single(state,'finish')
        state['status']='complete';state['point_goal_reached']=json.loads((ROOT/'result.json').read_text())['point_goal_reached']
    except BaseException:
        state['status']='failed';state['error']=traceback.format_exc()
        for child,log,f in active.values():stop(child);log.close()
        (ROOT/'FAILURE.txt').write_text(state['error']);raise
    finally:
        state['finished']=time.time();state['active']={};save(state)
        event('DONE',status=state['status'])

if __name__=='__main__':run()
