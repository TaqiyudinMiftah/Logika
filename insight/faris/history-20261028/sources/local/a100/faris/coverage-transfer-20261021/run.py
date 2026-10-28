"""Detached finite CPU campaign with an inherited deadline and retained models."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,time,subprocess,sys,fcntl,signal,traceback,shutil,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import baseline as b
def write(s):b.dump(s,ROOT/'state.json')
def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def main():
    lock=(ROOT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'plan.json').read_text());seal=json.loads((ROOT/'seal.json').read_text())
    for name,h in seal['files'].items():assert b.sha(ROOT/name)==h,name
    for name,h in seal['inputs'].items():assert b.sha(name)==h,name
    versions={name:importlib.metadata.version(name) for name in plan['versions']};assert versions==plan['versions']
    b.dump(versions,ROOT/'environment.json')
    state=json.loads((ROOT/'state.json').read_text()) if (ROOT/'state.json').exists() else dict(
        started=time.time(),deadline=plan['original_deadline'],completed_folds=[])
    assert state.get('status')!='complete' and state['deadline']==plan['original_deadline']
    for pid in state.get('active',{}).values():
        path=Path(f'/proc/{pid}/cmdline');assert not path.exists() or str(ROOT).encode() not in path.read_bytes()
    state['active']={};active={}
    def permitted():
        assert time.time()<state['deadline'] and not (ROOT/'STOP').exists()
        assert shutil.disk_usage(ROOT).free>10*1024**3
    def single(command,name):
        log=(ROOT/(name+'.log')).open('ab');p=subprocess.Popen([sys.executable,'-u',*command],stdout=log,stderr=log,start_new_session=True)
        state['active']={name:p.pid};write(state)
        try:
            while p.poll() is None:permitted();time.sleep(2)
            assert p.returncode==0,name+' failed; inspect its log'
        finally:stop(p);log.close();state['active']={};write(state)
    try:
        permitted();state['status']='building';write(state)
        if not (ROOT/'data/AUDIT.json').exists():single([str(ROOT/'source/build.py')],'build')
        state['status']='preflight';write(state)
        if not (ROOT/'preflight.json').exists():single([str(ROOT/'source/study.py'),'preflight'],'preflight')
        assert json.loads((ROOT/'preflight.json').read_text())['status']=='passed'
        pending=[]
        for f in range(5):
            d=ROOT/f'folds/fold{f}'
            if (d/'complete.json').exists():
                for name,h in json.loads((d/'complete.json').read_text())['files'].items():assert b.sha(d/name)==h
                if f not in state['completed_folds']:state['completed_folds'].append(f)
            else:
                if d.exists():
                    incident=ROOT/'incidents'/str(time.time_ns());incident.mkdir(parents=True);d.rename(incident/f'fold{f}')
                pending.append(f)
        state['status']='training';write(state)
        while pending or active:
            permitted()
            while pending and len(active)<5:
                f=pending.pop(0);log=(ROOT/f'fold{f}.log').open('ab')
                p=subprocess.Popen([sys.executable,'-u',str(ROOT/'source/study.py'),'fold','--fold',str(f)],
                    stdout=log,stderr=log,start_new_session=True);active[f]=(p,log)
                print(json.dumps(dict(stage='fold_started',fold=f,pid=p.pid)),flush=True)
            state['active']={str(f):p.pid for f,(p,_) in active.items()};write(state)
            for f,(p,log) in list(active.items()):
                if p.poll() is None:continue
                log.close();del active[f];assert p.returncode==0,f'Fold {f} failed'
                assert (ROOT/f'folds/fold{f}/complete.json').exists();state['completed_folds'].append(f)
                print(json.dumps(dict(stage='fold_complete',fold=f)),flush=True)
            time.sleep(2)
        state.update(status='reporting',active={});write(state)
        single([str(ROOT/'source/study.py'),'finish'],'finish')
        state.update(status='complete',finished=time.time());write(state)
        print(json.dumps(dict(stage='DONE',status='complete')),flush=True)
    except BaseException:
        for p,log in active.values():stop(p);log.close()
        state.update(status='failed',active={},error=traceback.format_exc());write(state);raise
if __name__=='__main__':main()
