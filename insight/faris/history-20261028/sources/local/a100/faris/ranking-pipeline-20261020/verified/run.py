"""Finite supervised driver; preserves the deadline and all produced models."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import concurrent.futures,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import baseline as b

def run():
    plan=json.loads((ROOT/'plan.json').read_text())
    assert not (ROOT/'state.json').exists(),'Existing attempt retained: never reset its clock'
    for name,h in json.loads((ROOT/'seal.json').read_text())['files'].items():
        assert b.sha(ROOT/name)==h,('changed input',name)
    started=plan['original_start'];state={'start':started,'deadline':started+plan['compute_ceiling_seconds'],'status':'training','completed_folds':[]}
    b.dump(state,ROOT/'state.json')
    def fold(f):
        with (ROOT/f'fold{f}.log').open('wb') as log:
            p=subprocess.run([sys.executable,str(ROOT/'source/pipeline.py'),'train','--root',str(ROOT),'--fold',str(f)],
                stdout=log,stderr=log,timeout=max(1,state['deadline']-time.time()),check=True)
        return f
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(fold,f) for f in range(5)]
            for future in concurrent.futures.as_completed(futures):
                state['completed_folds'].append(future.result());b.dump(state,ROOT/'state.json');print(json.dumps(state),flush=True)
        state['status']='evaluating';b.dump(state,ROOT/'state.json')
        subprocess.run([sys.executable,str(ROOT/'source/pipeline.py'),'evaluate','--root',str(ROOT)],check=True,timeout=max(1,state['deadline']-time.time()))
        subprocess.run([sys.executable,str(ROOT/'source/pipeline.py'),'package','--root',str(ROOT)],check=True,timeout=max(1,state['deadline']-time.time()))
        state['status']='complete';state['decision']=json.loads((ROOT/'decision.json').read_text())
    except BaseException:
        state['status']='failed';state['error']=traceback.format_exc();raise
    finally:
        state['finished']=time.time();b.dump(state,ROOT/'state.json');print(json.dumps(state),flush=True)

if __name__=='__main__':run()
