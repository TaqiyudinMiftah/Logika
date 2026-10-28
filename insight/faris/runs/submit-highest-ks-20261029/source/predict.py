"""One fixed production member; no model selection and no fitted-state writes."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse,json,time,traceback
import numpy as np
import pandas as pd
from common import R,P,read,dump,sha,utc,sources
from kumo_engine import predict

def main(fold,seed):
 import importlib.metadata as md
 import torch
 assert torch.cuda.is_available()
 config=read(R/'CONFIG.json');assert seed in config['seeds'] and fold in range(5)
 path=R/f'predictions/s{seed}/fold{fold}'
 assert not path.exists(),'Preserve earlier attempt; reconcile before explicit retry'
 path.mkdir(parents=True)
 start=time.time()
 try:
  manifest=read(R/f'packets/fold{fold}/MANIFEST.json');packet=R/f'packets/fold{fold}'
  for filename,key in [('features.parquet','feature_sha256'),('y_context.npy','labels_sha256'),('test_ids.npy','test_ids_sha256')]:assert sha(packet/filename)==manifest[key]
  parameters=read(P/'CONFIG.json')['designs']['kumo_large_global99_e2']['parameters']
  # The selected recipe was registered in the original CONFIG.
  frame=pd.read_parquet(packet/'features.parquet');y=np.load(packet/'y_context.npy');ids=np.load(packet/'test_ids.npy')
  columns=manifest['columns'];assert frame.shape==(100000,99) and len(y)==70000 and len(ids)==30000
  checkpoint=read(P/'PATHS.json')['checkpoint'];assert sha(checkpoint)==config['pretrained_sha256']
  metadata=dict(status='running',UTC=utc(),fold=fold,base_seed=seed,effective_seed=seed+fold,
       columns=columns,context_rows=70000,test_rows=30000,parameters=parameters,
       packet_manifest_sha256=sha(packet/'MANIFEST.json'),source_sha256=sources(),
       GPU=torch.cuda.get_device_name(0),GPU_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
       versions={k:md.version(k)for k in ['torch','structured-data-models','numpy','pandas','scipy','scikit-learn']},
       CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),no_query_labels=True,
       fitted_weights_saved=False,KV_caches_saved=False)
  dump(metadata,path/'START.json')
  raw,audit=predict(frame.iloc[:70000],frame.iloc[70000:],columns,y,checkpoint,seed+fold,
        engine=parameters.get('engine','concat_context_vocab'),estimators=int(parameters.get('estimators',parameters.get('num_estimators',2))),
        pred_batch=int(parameters.get('prediction_batch',256)),mem_fraction=float(parameters.get('memory_fraction',.7)),
        size='large',expected_sha=config['pretrained_sha256'])
  assert raw.shape==(30000,) and np.isfinite(raw).all() and np.all((raw>=0)&(raw<=1))
  np.savez_compressed(path/'prediction.npz',application_id=ids,raw=raw)
  dump(dict(status='complete',UTC=utc(),seconds=time.time()-start,fold=fold,base_seed=seed,
        effective_seed=seed+fold,audit=audit,start_sha256=sha(path/'START.json'),prediction_sha256=sha(path/'prediction.npz'),
        fitted_weights_saved=False,KV_caches_saved=False),path/'RESULT.json')
  print(json.dumps(dict(status='complete',fold=fold,seed=seed,seconds=time.time()-start)),flush=True)
 except BaseException:
  dump(dict(status='failed',UTC=utc(),seconds=time.time()-start,diagnostic=traceback.format_exc(),
       resource_failure_is_not_predictive_evidence=True),path/'FAILURE.json');raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,required=True);parser.add_argument('--seed',type=int,required=True);a=parser.parse_args();main(a.fold,a.seed)
