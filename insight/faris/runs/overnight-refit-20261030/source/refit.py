"""Fresh current-campaign CPU CANN / GPU Kumo production member; weights stay in RAM."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,hashlib,json,time,traceback
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,default=str)+'\n');q.replace(p)
def run(kind,seed,fold):
 clock=json.load(open(R/'CLOCK.json'));assert time.time()<clock['admission_end_epoch'];d=R/f'predictions/{kind}/s{seed}/fold{fold}';d.mkdir(parents=True,exist_ok=False);start=time.time()
 try:
  packet=R/f'packets/fold{fold}';m=json.load(open(packet/'MANIFEST.json'))
  for n,key in [('features.parquet','feature_sha256'),('y.npy','labels_sha256'),('test_ids.npy','test_ids_sha256')]:assert sha(packet/n)==m[key]
  columns=m['views']['actuarial99'if kind=='cann'else'global99'];frame=pd.read_parquet(packet/'features.parquet',columns=columns);y=np.load(packet/'y.npy');ids=np.load(packet/'test_ids.npy');assert frame.shape==(100000,99)and len(y)==70000 and len(ids)==30000
  dump(d/'START.json',dict(epoch=start,kind=kind,base_seed=seed,seed=seed+fold,fold=fold,context_rows=70000,query_rows=30000,no_query_labels=True,columns=columns,packet_manifest_sha256=sha(packet/'MANIFEST.json'),source_sha256={p.name:sha(p)for p in(R/'source').glob('*.py')},campaign='overnight-refit-20261030',fresh_training=True))
  if kind=='cann':
   from cpu_models import train_predict
   spec=dict(offset_mode='intercept',glm_C=.1,width=128,residual_penalty=0.,learning_rate=.0003,deadline_epoch=clock['model_deadline_epoch']-60,max_seconds=1800,inner_split_seed=20261029+fold)
   raw,audit=train_predict(frame,np.arange(70000),y,np.arange(70000,100000),spec,seed+fold)
  else:
   from kumo_engine import predict
   from context_recipe import sample_context
   env=json.load(open(R/'ENVIRONMENT.json'));checkpoint=env['checkpoint_large'];expected='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
   count=25000 if kind=='kumo25'else 70000;selected,draw=(np.arange(70000),None) if count==70000 else sample_context(y,count,seed+fold,0)
   raw,audit=predict(frame.iloc[selected],frame.iloc[70000:],columns,y[selected],checkpoint,seed+fold,engine='concat_context_vocab',estimators=2,pred_batch=256,mem_fraction=.7,size='large',expected_sha=expected)
   audit.update(context_count=count,context_draw_seed=draw,context_rows_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest())
  assert raw.shape==(30000,)and np.isfinite(raw).all()and np.all((raw>=0)&(raw<=1))
  with(d/'prediction.npz.tmp').open('wb')as f:np.savez_compressed(f,raw=raw,application_id=ids)
  (d/'prediction.npz.tmp').replace(d/'prediction.npz')
  dump(d/'RESULT.json',dict(status='complete',epoch=time.time(),seconds=time.time()-start,kind=kind,fold=fold,base_seed=seed,audit=audit,start_sha256=sha(d/'START.json'),prediction_sha256=sha(d/'prediction.npz'),fitted_weights_saved=False,KV_caches_saved=False,fresh_training=True,campaign='overnight-refit-20261030'))
  print(json.dumps(dict(kind=kind,seed=seed,fold=fold,status='complete',seconds=time.time()-start)),flush=True)
 except BaseException:
  dump(d/'FAILURE.json',dict(status='failed',epoch=time.time(),diagnostic=traceback.format_exc(),no_automatic_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['cann','kumo25','kumo70'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True);a=p.parse_args();run(a.kind,a.seed,a.fold)
