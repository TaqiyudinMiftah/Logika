"""Compact supervised models; no evaluation labels enter training or retrieval."""
import copy,math
import numpy as np
import torch
from torch import nn

class Residual(nn.Module):
 def __init__(self,d,w,blocks,drop):
  super().__init__();self.input=nn.Linear(d,w);self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w*2),nn.ReLU(),nn.Dropout(drop),nn.Linear(w*2,w)) for _ in range(blocks)]);self.out=nn.Linear(w,1)
 def forward(self,x):
  h=self.input(x)
  for b in self.blocks:h=h+b(h)
  return self.out(h)
class FT(nn.Module):
 def __init__(self,d,w,blocks,drop):
  super().__init__();w=min(w,128);self.weight=nn.Parameter(torch.randn(d,w)*.02);self.bias=nn.Parameter(torch.zeros(d,w));self.cls=nn.Parameter(torch.zeros(1,1,w));self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(w,4,w*2,drop,batch_first=True,norm_first=True),blocks);self.out=nn.Linear(w,1)
 def forward(self,x):
  h=x[:,:,None]*self.weight[None]+self.bias[None];h=torch.cat([self.cls.expand(len(x),-1,-1),h],dim=1);return self.out(self.enc(h)[:,0])
class History(nn.Module):
 def __init__(self,d,w,blocks,drop,kind):
  super().__init__();self.kind=kind;self.event=nn.Sequential(nn.Linear(9,w),nn.ReLU());self.gru=nn.GRU(w,w,batch_first=True) if kind=='gru' else None;self.head=Residual(d+w,w,blocks,drop)
 def forward(self,x,event,lengths):
  h=self.event(event)
  if self.gru:
   packed=nn.utils.rnn.pack_padded_sequence(h,lengths.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False);_,s=self.gru(packed);v=s[-1]
  else:
   mask=torch.arange(h.shape[1],device=h.device)[None,:]<lengths[:,None];v=(h*mask[:,:,None]).sum(1)/lengths.clamp_min(1)[:,None]
  v=v*(lengths>0)[:,None];return self.head(torch.cat([x,v],1))

def make(kind,d,p,x):
 w=p.get('width',128);blocks=p.get('blocks',2);drop=p.get('dropout',.1)
 if kind=='tabm':
  from tabm import TabM
  emb=None
  if p.get('encoding')=='ple':
   from rtdl_num_embeddings import PiecewiseLinearEmbeddings,compute_bins
   emb=PiecewiseLinearEmbeddings(compute_bins(torch.tensor(x),n_bins=16),d_embedding=8,activation=True,version='B')
  return TabM.make(n_num_features=d,d_out=1,k=16,n_blocks=blocks,d_block=w,dropout=drop,num_embeddings=emb)
 if kind=='ft':return FT(d,w,blocks,drop)
 if kind in ['gru','deepsets']:return History(d,w,blocks,drop,kind)
 return Residual(d,w,blocks,drop)

def train(ctx,kind,x,y,q,folder,p,events=None,query_events=None,valid=None,epochs=None,seed=2026):
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(ctx.spec['resources']['cpus']);device='cuda' if ctx.spec['resources'].get('gpus') else 'cpu'
 model=make(kind,x.shape[1],p,x).to(device);opt=torch.optim.AdamW(model.parameters(),lr=p.get('lr',.0003),weight_decay=.01)
 xx=torch.as_tensor(x,device=device);yy=torch.as_tensor(y,dtype=torch.float32,device=device);qq=torch.as_tensor(q,device=device)
 ee=ll=qe=ql=None
 if events is not None:
  ee=torch.as_tensor(events[0],device=device);ll=torch.as_tensor(events[1],device=device);qe=torch.as_tensor(query_events[0],device=device);ql=torch.as_tensor(query_events[1],device=device)
 batch=128 if kind in ['ft','gru','deepsets'] else 512
 def logits(a,e=None,l=None):
  z=model(a,e,l) if kind in ['gru','deepsets'] else model(a)
  return z.squeeze(-1)
 @torch.no_grad()
 def infer():
  model.eval();values=[]
  for start in range(0,len(qq),batch*2):
   z=logits(qq[start:start+batch*2],qe[start:start+batch*2] if qe is not None else None,ql[start:start+batch*2] if ql is not None else None).sigmoid()
   if z.ndim==2:z=z.mean(1)
   values.append(z.float().cpu().numpy())
  return np.concatenate(values)
 best=float('inf');best_epoch=1;stale=0;losses=[];weight=p.get('weight',1)
 for epoch in range(1,(epochs or 100)+1):
  ctx.check_stop();model.train();order=torch.randperm(len(xx),device=device)
  for ix in order.split(batch):
   opt.zero_grad(set_to_none=True);z=logits(xx[ix],ee[ix] if ee is not None else None,ll[ix] if ll is not None else None);target=yy[ix,None].expand_as(z) if z.ndim==2 else yy[ix]
   loss=nn.functional.binary_cross_entropy_with_logits(z,target,pos_weight=torch.tensor(float(weight),device=device));loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if valid is not None:
   pred=np.clip(infer(),1e-7,1-1e-7);current=float(-(weight*valid*np.log(pred)+(1-valid)*np.log(1-pred)).mean());losses.append(current)
   if current<best-1e-5:best=current;best_epoch=epoch;stale=0
   else:stale+=1
   if stale>=10:break
  if epoch%10==0:
   name=folder+f'/epoch_{epoch}.pt';torch.save(model.state_dict(),ctx.path(name));ctx.model(name)
  print('epoch',epoch,'inner_loss',losses[-1] if losses else None,flush=True)
 name=folder+'/weights.pt';torch.save({'state':model.cpu().state_dict(),'kind':kind,'params':p,'features':x.shape[1],'epochs':epoch,'seed':seed},ctx.path(name));ctx.model(name);model.to(device)
 pred=infer();ctx.save_json(folder+'/training.json',{'epochs':epoch,'selected_epoch':best_epoch,'inner_losses':losses,'device':device,'model':repr(model),'implementation':'TabM official package; FT, residual, histories implemented locally; retrieval_mlp is TabR-inspired, not an exact TabR reproduction'})
 del model,opt,xx,yy,qq
 if device=='cuda':torch.cuda.empty_cache()
 return np.clip(pred/(weight*(1-pred)+pred),1e-8,1-1e-8),best_epoch
