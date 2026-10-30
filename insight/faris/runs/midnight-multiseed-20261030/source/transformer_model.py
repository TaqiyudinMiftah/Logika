"""FT-inspired tabular Transformer: scalar CLS versus LocalGLM-style coefficient head.
This is a documented adaptation, not an exact reproduction of FT-Transformer/BERT.
"""
import os,time,gc,hashlib
import numpy as np,pandas as pd,torch
from scipy.special import logit
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
class Encoder:
 def fit(self,frame,rows):
  self.maps=[];self.cards=[]
  for col in frame:
   x=frame[col].iloc[rows]
   if pd.api.types.is_numeric_dtype(x):
    v=x.to_numpy(dtype=float,na_value=np.nan);v=np.where(np.isfinite(v),v,np.nan);median=float(np.nanmedian(v))if np.isfinite(v).any()else 0.;v=np.nan_to_num(v,nan=median);self.maps.append(('num',median,float(v.mean()),max(float(v.std()),1e-6)));self.cards.append(0)
   else:
    values=sorted(x.dropna().astype(str).unique());self.maps.append(('cat',{v:i+1 for i,v in enumerate(values)}));self.cards.append(len(values)+1)
  self.rows_sha256=hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest();return self
 def transform(self,frame,rows):
  out=[]
  for col,m in zip(frame,self.maps):
   x=frame[col].iloc[rows]
   if m[0]=='num':v=x.to_numpy(dtype=float,na_value=np.nan);v=np.where(np.isfinite(v),v,m[1]);out.append(np.clip((v-m[2])/m[3],-10,10))
   else:out.append(x.astype('string').map(m[1]).fillna(0).to_numpy(dtype=float))
  return torch.as_tensor(np.column_stack(out),dtype=torch.float32,device='cuda')
class Net(torch.nn.Module):
 def __init__(self,cards,coefficient):
  super().__init__();d=16;self.cards=cards;self.coefficient=coefficient;self.weight=torch.nn.Parameter(torch.randn(len(cards),d)*.02);self.bias=torch.nn.Parameter(torch.zeros(len(cards),d));self.emb=torch.nn.ModuleDict({str(i):torch.nn.Embedding(c,d)for i,c in enumerate(cards)if c});self.cls=torch.nn.Parameter(torch.randn(1,1,d)*.02);self.layer=torch.nn.TransformerEncoderLayer(d,4,64,.1,activation='gelu',batch_first=True,norm_first=True);self.norm=torch.nn.LayerNorm(d);self.head=torch.nn.Linear(d,1);torch.nn.init.zeros_(self.head.weight);torch.nn.init.zeros_(self.head.bias)
 def forward(self,x):
  tok=x.unsqueeze(-1)*self.weight[None]+self.bias[None]
  for i,c in enumerate(self.cards):
   if c:tok[:,i]=self.emb[str(i)](x[:,i].long())+self.bias[i]
  h=self.norm(self.layer(torch.cat([self.cls.expand(len(x),-1,-1),tok],dim=1)))
  if not self.coefficient:return self.head(h[:,0]).squeeze(-1)
  scale=x.clone()
  for i,c in enumerate(self.cards):
   if c:scale[:,i]=1.
  return (self.head(h[:,1:]).squeeze(-1)*scale).sum(-1)/(len(self.cards)**.5)
def train_predict(frame,ctx,y,query,spec,seed):
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.75)
 y=np.asarray(y,np.int64);ctx=np.asarray(ctx,np.int64);start=time.time();end=min(spec['deadline_epoch'],start+spec['max_seconds']);inner,stop=train_test_split(np.arange(len(ctx)),test_size=.15,stratify=y,random_state=spec['inner_split_seed']);enc=Encoder().fit(frame,ctx[inner]);x=enc.transform(frame,ctx);xx=enc.transform(frame,query);yy=torch.as_tensor(y,dtype=torch.float32,device='cuda');tr=torch.as_tensor(inner,device='cuda');va=torch.as_tensor(stop,device='cuda');batch=256
 def model():torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);return Net(enc.cards,spec['coefficient']).cuda()
 def pred(m,z,offset,b=512):
  m.eval()
  with torch.no_grad():return torch.cat([torch.sigmoid(offset+m(q))for q in z.split(b)]).cpu().numpy().astype(float)
 def epoch(m,op,index,offset,g):
  assert time.time()<end;m.train();order=torch.randperm(len(index),generator=g)
  for block in order.split(batch):
   assert time.time()<end;idx=index[block.to('cuda')];op.zero_grad(set_to_none=True);r=m(x[idx]);loss=torch.nn.functional.binary_cross_entropy_with_logits(offset+r,yy[idx])+.1*r.square().mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);op.step()
 m=model();op=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);g=torch.Generator().manual_seed(seed+271828);offset=float(logit(y[inner].mean()));best=0;bestks=float(ks_2samp(pred(m,x[va],offset)[y[stop]==1],pred(m,x[va],offset)[y[stop]==0]).statistic);stale=0
 for e in range(1,41):
  epoch(m,op,tr,offset,g);p=pred(m,x[va],offset);score=float(ks_2samp(p[y[stop]==1],p[y[stop]==0]).statistic)
  if score>bestks+1e-12:best,bestks,stale=e,score,0
  else:stale+=1
  if stale>=6:break
 del m,op;m=model();op=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);g=torch.Generator().manual_seed(seed+271828);allrows=torch.arange(len(ctx),device='cuda');offset=float(logit(y.mean()))
 for e in range(best):epoch(m,op,allrows,offset,g)
 p=pred(m,xx,offset);replay=pred(m,xx[:128],offset,64);error=float(np.max(abs(p[:128]-replay)));assert error<=5e-6
 audit=dict(model_family='FTinspiredLocalGLMcoeff'if spec['coefficient']else'FTinspiredCLScontrol',width=16,layers=1,heads=4,FFN=64,dropout=.1,batch=256,max_epochs=40,patience=6,selected_epochs=best,encoder_fit_rows=len(inner),encoder_fit_rows_sha256=enc.rows_sha256,stop_rows=len(stop),context_rows=len(ctx),final_refit_rows=len(ctx),inner_encoder_retained=True,query_labels_used=False,batch_replay_max_abs_probability_error=error,fitted_weights_saved=False,KV_caches_saved=False,seed=seed,CPU_initialization_CUDA_dropout=True,seconds=time.time()-start)
 del m,op,x,xx,yy;gc.collect();torch.cuda.empty_cache();return p,audit
