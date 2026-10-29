"""Run sealed model code on an explicitly preregistered additional backend.

Pilot whitelist extension is limited to the published resource addendum. The
original CONFIG/source/clock are unchanged. Only prediction-file I/O is atomic.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import check_seal,load,sha,catalog
import numpy as np
a=argparse.ArgumentParser();a.add_argument('--task',required=True);a.add_argument('--owner',required=True);a.add_argument('--pilot',action='store_true');v=a.parse_args()
check_seal();plan=load('COLAB_RESOURCE_ADDENDUM.json');receipt=load('RESOURCE_PREPUBLICATION.json')
assert receipt['extra_designs_sha256']==sha(R/'EXTRA_DESIGNS.jsonl') and receipt['source_seal_sha256']==sha(R/'SOURCE_SEAL.json')
owner=plan['assignments'][v.owner];assert v.task.split('__s')[0]in owner['designs']
if v.pilot:assert v.task in owner['pilot_tasks']
original_np=np.savez_compressed
def atomic(file,*args,**kwargs):
    if isinstance(file,(str,bytes,os.PathLike))and Path(file).name=='prediction.npz':
        p=Path(file);tmp=p.with_name('prediction.'+str(os.getpid())+'.atomic-tmp')
        with tmp.open('wb')as f:original_np(f,*args,**kwargs)
        tmp.replace(p);return
    return original_np(file,*args,**kwargs)
np.savez_compressed=atomic
spec=importlib.util.spec_from_file_location('sealed_model_worker',R/'source/worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_load=m.load
def authorized_load(name):
    result=original_load(name)
    if v.pilot and name=='CONFIG.json':
        result=copy.deepcopy(result);result['pilot_tasks']=result['pilot_tasks']+owner['pilot_tasks']
    return result
m.load=authorized_load;m.run(v.task,attempt=1,pilot=v.pilot)
