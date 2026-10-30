"""Exact frozen-branch replication. Fresh models stay in RAM; save compact predictions only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ.setdefault('SDM_CHUNK_MEMORY_FRACTION','.02')
from pathlib import Path
import argparse,json,time,hashlib,sys,traceback,socket,resource
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
SHA_LARGE='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,default=str)+'\n');t.replace(p)
def fit(kind,seed,fold,phase,reference=False):
 C=json.loads((R/'CLOCK.json').read_text());assert time.time()<C['admission_end_epoch'];assert (R/'PREPUBLICATION_SOURCE.json').exists();ident=('ref_'if reference else'')+f'{kind}__s{seed}__f{fold}';out=R/'runs'/ident/phase
 assert not out.exists(),'Attempt alreadyexists;never automaticretry';out.mkdir(parents=True);start=time.time()
 try:
  for n,h in json.loads((R/'SOURCE_SEAL.json').read_text()).items():assert sha(R/n)==h
  p=R/f'packets/fold{fold}';m=json.loads((p/'MANIFEST.json').read_text());cols=m['views']['global99'if kind=='kumo70'else'actuarial99'];assert len(cols)==99
  filename=('localglm' if kind.startswith('transformer_') else kind)+'_validation.parquet' if phase=='validation' else 'features.parquet'
  for n in [filename,'roles.npz']:assert sha(p/n)==m['files'][n]
  frame=pd.read_parquet(p/filename,columns=cols);roles=np.load(p/'roles.npz');assert frame.shape==((70000 if phase=='validation' else 100000),99)
  if phase=='validation':ctx=roles['context'];query=np.r_[roles['correction'],roles['validation']];y=roles['y_context'].astype(np.int64);assert len(ctx)==50000 and len(query)==20000
  else:assert sha(p/'y.npy')==m['files']['y.npy'];ctx=np.arange(70000);query=np.arange(70000,100000);y=np.load(p/'y.npy').astype(np.int64)
  assert not np.intersect1d(ctx,query).size and len(y)==len(ctx)
  dump(out/'START.json',dict(epoch=start,kind=kind,phase=phase,base_seed=seed,seed=seed+fold,fold=fold,reference=reference,host=socket.gethostname(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),context_rows=len(ctx),query_rows=len(query),context_sha256=hashlib.sha256(ctx.astype('<i8').tobytes()).hexdigest(),query_sha256=hashlib.sha256(query.astype('<i8').tobytes()).hexdigest(),packet_sha256=sha(p/'MANIFEST.json'),config_sha256=sha(R/'CONFIG.json'),clock_sha256=sha(R/'CLOCK.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),source_sha256=sha(__file__),query_labels_loaded=False,fresh_training=True))
  if kind=='kumo70':
   sys.path.insert(0,str(R/'source'));from kumo_engine import predict
   checkpoint=R/'checkpoints/large.pt';assert sha(checkpoint)==SHA_LARGE
   raw,audit=predict(frame.iloc[ctx],frame.iloc[query],cols,y,str(checkpoint),seed+fold,engine='concat_context_vocab',estimators=2,pred_batch=256,mem_fraction=.7,size='large',expected_sha=SHA_LARGE)
  elif kind.startswith('transformer_'):
   sys.path.insert(0,str(R/'source'));from transformer_model import train_predict
   spec=dict(coefficient=kind=='transformer_coefficient',inner_split_seed=20261029+fold,deadline_epoch=C['model_deadline_epoch']-30,max_seconds=min(1800,C['model_deadline_epoch']-time.time()-30));raw,audit=train_predict(frame,ctx,y,query,spec,seed+fold)
  else:
   os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(R/'source'/f'{kind}_{phase}'));from cpu_models import train_predict
   spec=dict(offset_mode='intercept',glm_C=.1,width=64 if kind=='localglm' else 128,residual_penalty=.1 if kind=='localglm' else 0.,learning_rate=.0003,book_transform='raw',deadline_epoch=C['model_deadline_epoch']-30,max_seconds=min(1800,C['model_deadline_epoch']-time.time()-30),inner_split_seed=20261029+fold)
   raw,audit=train_predict(frame,ctx,y,query,spec,seed+fold)
  assert raw.shape==(len(query),)and np.isfinite(raw).all()and np.all((raw>=0)&(raw<=1))
  if kind=='kumo70':assert audit['first_batch_replay_error']<1e-7
  else:assert audit['batch_replay_max_abs_probability_error']<=5e-6
  referror=None
  if reference:
   old=np.load(R/f'reference/{kind}_{phase}.npz')['raw'];referror=float(np.max(abs(raw-old)));assert kind=='kumo70'or referror<=5e-6
  with(out/'prediction.tmp').open('wb')as f:np.savez_compressed(f,rows=query,raw=raw)
  (out/'prediction.tmp').replace(out/'prediction.npz');dump(out/'RESULT.json',dict(status='complete',kind=kind,phase=phase,base_seed=seed,fold=fold,reference=reference,seconds=time.time()-start,prediction_sha256=sha(out/'prediction.npz'),start_sha256=sha(out/'START.json'),audit=audit,reference_max_abs=referror,crossGPU_reference_difference_diagnostic_only=kind=='kumo70',max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,fitted_weights_saved=False,KV_cache_saved=False));print(json.dumps(dict(complete=ident,phase=phase,seconds=time.time()-start)),flush=True)
 except BaseException:dump(out/'FAILURE.json',dict(epoch=time.time(),diagnostic=traceback.format_exc(),no_automatic_retry=True));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--phase',choices=['validation','production'],required=True);p.add_argument('--reference',action='store_true');a=p.parse_args();fit(a.kind,a.seed,a.fold,a.phase,a.reference)
