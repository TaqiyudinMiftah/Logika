"""Offline GPU foundation inference and bounded training-only episodic adaptation."""
import gc,os
from pathlib import Path
import numpy as np,joblib
from features import Processor
from utils import sha

def fit_foundation(ctx,bank,p,rep,rows,y,query,folder,seed):
 import torch
 torch.set_num_threads(ctx.spec['resources']['cpus']);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(min(.90,ctx.spec['resources']['vram_mb']*1048576/torch.cuda.get_device_properties(0).total_memory))
 pre=Processor(bank,rep,'lgb',seed);x=pre.fit_transform(rows,y);q=pre.transform(query);pre.detach();name=folder+'/preprocessing.joblib';joblib.dump(pre,ctx.path(name),compress=1);ctx.model(name)
 from sklearn.model_selection import train_test_split
 n=min(len(rows),p.get('context',32768));ci=np.arange(len(rows)) if n==len(rows) else train_test_split(np.arange(len(rows)),test_size=n,stratify=y,random_state=seed)[1]
 np.save(ctx.path(folder+'/context_rows.npy'),rows[ci]);ctx.model(folder+'/context_rows.npy')
 if p['model']=='tabicl':
  from tabicl import TabICLClassifier
  checkpoint=Path(ctx.inputs['weights'])/'tabicl-classifier-v2-20260212.ckpt';assert checkpoint.exists()
  m=TabICLClassifier(n_estimators=p.get('estimators',4),model_path=str(checkpoint),allow_auto_download=False,device='cuda',n_jobs=ctx.spec['resources']['cpus'],batch_size=1,random_state=seed,kv_cache=True,offload_mode='auto',disk_offload_dir=str(ctx.path(folder+'/offload')))
  m.fit(x[ci],y[ci]);values=m.predict_proba(q)[:,1];name=folder+'/fitted.pkl';m.save(str(ctx.path(name)),save_model_weights=False,save_training_data=True,save_kv_cache=False);ctx.model(name)
 else:
  import pyarrow as pa
  from sdm import TableTensor
  from sdm.models import KumoTabular
  from kumo_cuda_cache import keep_fit_cache_on_cuda
  def tensor(a):
   cols={f'f{i}':pa.array(a[:,i],type=pa.float64()) for i in range(a.shape[1])};old=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
   try:return TableTensor.from_arrow(pa.table(cols),stypes={k:'numerical' for k in cols},device='cuda')
   finally:torch.set_default_dtype(old)
  def labels(a):return TableTensor.from_arrow(pa.table({'target':pa.array(a.astype(np.int64),type=pa.int64())}),stypes={'target':'categorical'},device='cuda')
  m=KumoTabular(task='classification',size='large',pretrained=True,device='cuda');keep_fit_cache_on_cuda(m);updates=p.get('updates',0);rng=np.random.default_rng(seed)
  if updates:
   m.train();opt=torch.optim.AdamW(m.parameters(),lr=p.get('lr',1e-6),weight_decay=.01);losses=[]
   for step in range(updates):
    ctx.check_stop();ids=rng.choice(len(rows),160,replace=False);a,b=ids[:128],ids[128:];assert not np.intersect1d(a,b).size;opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda',dtype=torch.bfloat16):
     out=m(tensor(x[a]),labels(y[a]),tensor(x[b]),num_estimators=1,estimator_batch_size=1,generator=torch.Generator(device='cuda').manual_seed(seed+step));prob=out[['1']].numerical.squeeze(-1).float().clamp(1e-6,1-1e-6);assert prob.requires_grad
    loss=torch.nn.functional.binary_cross_entropy(prob,torch.tensor(y[b],dtype=torch.float32,device='cuda'));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    if (step+1)%16==0:name=folder+f'/updates_{step+1}.pt';torch.save(m.state_dict(),ctx.path(name));ctx.model(name)
   ctx.save_json(folder+'/adaptation.json',{'losses':losses,'training_only':True});del opt
  # Identical frozen checkpoints are hard-linked, with an explicit content identity.
  store=ctx.campaign.parent/'shared-weights';store.mkdir(exist_ok=True)
  name=folder+'/weights.pt'
  if updates:torch.save(m.state_dict(),ctx.path(name))
  else:
   target=store/'kumo-large-frozen.pt'
   if not target.exists():
    temp=target.with_suffix('.'+str(os.getpid())+'.tmp');torch.save(m.state_dict(),temp);os.replace(temp,target)
   os.link(target,ctx.path(name))
  ctx.model(name);m.eval()
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16):m.fit(tensor(x[ci]),labels(y[ci]),num_estimators=p.get('estimators',4),estimator_batch_size=1,generator=torch.Generator(device='cuda').manual_seed(seed))
  import cloudpickle
  name=folder+'/fitted_context.pkl'
  with ctx.path(name).open('wb') as f:cloudpickle.dump(m._cache['recipe_execution'],f)
  ctx.model(name);values=[]
  for start in range(0,len(q),512):
   ctx.check_stop()
   with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16):out=m.predict(tensor(q[start:start+512]))
   values.append(out[['1']].numerical.squeeze(-1).float().cpu().numpy())
  values=np.concatenate(values)
 ctx.save_json(folder+'/foundation.json',{'model':p['model'],'context_count':n,'updates':p.get('updates',0),'gpu':torch.cuda.get_device_name(),'peak_vram_bytes':torch.cuda.max_memory_allocated(),'preprocessing':pre.audit})
 del m,x,q;gc.collect();torch.cuda.empty_cache();return np.clip(values,1e-8,1-1e-8)
