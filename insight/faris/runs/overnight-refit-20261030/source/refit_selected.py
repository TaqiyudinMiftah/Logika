"""Frozen study production adapter; no selection or fitted-state serialization."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,hashlib,json,sys,time,traceback
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,default=str)+'\n');q.replace(p)
def run(namespace,seed,fold):
 config=json.load(open(R/'selected_fullfits'/namespace/'SELECTION.json'));study=R/namespace;design=config['design'];spec=dict(config['spec']);kind=config['kind'];folder=R/f'predictions/{kind}/s{seed}/fold{fold}';folder.mkdir(parents=True,exist_ok=False);start=time.time()
 try:
  m=json.load(open(R/f'packets/fold{fold}/MANIFEST.json'));packet=R/f'packets/fold{fold}';columns=m['views'][spec['view']];assert columns==json.load(open(study/f'packets/fold{fold}/MANIFEST.json'))['views'][spec['view']]
  frame=pd.read_parquet(packet/'features.parquet',columns=columns);y=np.load(packet/'y.npy');ids=np.load(packet/'test_ids.npy');assert frame.shape==(100000,99)
  source=R/'selected_fullfits'/namespace/'source';sys.path.insert(0,str(source));clock=json.load(open(R/'CLOCK.json'));assert time.time()<clock['admission_end_epoch']
  spec.update(deadline_epoch=clock['model_deadline_epoch']-60,max_seconds=1200,inner_split_seed=20261029+fold)
  dump(folder/'START.json',dict(epoch=start,campaign='overnight-refit-20261030',fresh_training=True,namespace=namespace,design=design,spec=spec,seed=seed+fold,base_seed=seed,fold=fold,context_rows=70000,query_rows=30000,no_query_labels=True,production_packet_sha256=sha(packet/'MANIFEST.json'),source_sha256={p.name:sha(p)for p in source.glob('*.py')}))
  if spec['resource']=='cpu':
   from cpu_models import train_predict
   raw,audit=train_predict(frame,np.arange(70000),y,np.arange(70000,100000),spec,seed+fold)
  else:
   from gpu_models import train_predict
   raw,audit=train_predict(frame,np.arange(70000),y,np.arange(70000,100000),spec,seed+fold,device='cuda')
  assert raw.shape==(30000,)and np.isfinite(raw).all()and np.all((raw>=0)&(raw<=1))
  with(folder/'prediction.tmp').open('wb')as f:np.savez_compressed(f,raw=raw,application_id=ids)
  (folder/'prediction.tmp').replace(folder/'prediction.npz');dump(folder/'RESULT.json',dict(status='complete',epoch=time.time(),seconds=time.time()-start,campaign='overnight-refit-20261030',fresh_training=True,fold=fold,base_seed=seed,start_sha256=sha(folder/'START.json'),prediction_sha256=sha(folder/'prediction.npz'),audit=audit,fitted_weights_saved=False,KV_caches_saved=False))
 except BaseException:dump(folder/'FAILURE.json',dict(epoch=time.time(),diagnostic=traceback.format_exc(),no_automatic_retry=True));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--namespace',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--fold',type=int,required=True);p=a.parse_args();run(p.namespace,p.seed,p.fold)
