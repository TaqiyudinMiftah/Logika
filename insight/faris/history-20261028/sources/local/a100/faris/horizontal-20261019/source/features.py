"""Read-only old bank plus point-in-time feature blocks and train-only transforms."""
import copy,json
from pathlib import Path
import numpy as np,pandas as pd
from metaexp.features import Bank
from metaexp.pipeline import Preprocessor

class ExploreBank(Bank):
 def __init__(self,root,extras):
  super().__init__(root);self.extra_root=Path(extras)
  self.extra_meta=json.loads((self.extra_root/'manifest.json').read_text());self.extra=np.load(self.extra_root/'numeric.npy',mmap_mode='r')
  from utils import ident
  assert self.extra_meta['base_bank_id']==self.manifest['bank_id']
  assert self.extra_meta['rows_sha256']==ident(self.metadata.application_id.tolist())
  self.extra_names=self.extra_meta['columns'];self.extra_lookup={n:i for i,n in enumerate(self.extra_names)}
  self.manifest=copy.deepcopy(self.manifest)
  for n,inf in self.extra_meta['info'].items():self.manifest['info'][n]=inf
 def columns(self,rep):
  r=copy.deepcopy(rep);names,cats=super().columns(r);c=rep['config'];family=c.get('_extra_family')
  if family and family!='representation':
   names += [n for n in self.extra_names if self.extra_meta['info'][n]['family']==family and self.extra_meta['info'][n]['window'] in ['all',c['window']] and self.extra_meta['info'][n].get('half_life',c['half_life'])==c['half_life'] and (not c.get('_extra_columns') or n in c['_extra_columns'])]
  return list(dict.fromkeys(names)),cats
 def cohort(self,fitrows,rows,names,c):
  original=[n for n in names if n not in self.extra_lookup];extra=[n for n in names if n in self.extra_lookup]
  x,n=super().cohort(fitrows,rows,original,c)
  if extra:x=np.column_stack([x,np.array(self.extra[np.asarray(rows)[:,None],[self.extra_lookup[v] for v in extra]],dtype=np.float32)]);n+=extra
  added=[];added_names=[]
  for name in extra:
   if not name.endswith('_k') or name[:-2]+'_n' not in extra:continue
   kn=self.extra_lookup[name];nn=self.extra_lookup[name[:-2]+'_n'];k=self.extra[fitrows,kn];count=self.extra[fitrows,nn];prior=float(np.nansum(k)/max(np.nansum(count),1));strength=c['strength'];kr=self.extra[rows,kn];nr=self.extra[rows,nn];rate=np.divide(kr+strength*prior,nr+strength,out=np.full(len(rows),np.nan,dtype=np.float32),where=nr+strength>0);added.extend([rate,np.sqrt(np.maximum(rate*(1-rate)/(nr+strength+1),0))]);added_names += [name+'_posterior',name+'_uncertainty']
  if added:
   for name in added_names:self.manifest['info'][name]={'sources':['repayment']}
   x=np.column_stack([x,*added]);n+=added_names
  return x,n

class Processor:
 def __init__(self,bank,rep,model,seed):self.bank=bank;self.rep=copy.deepcopy(rep);self.model=model;self.seed=seed
 def fit_transform(self,rows,y):
  r=copy.deepcopy(self.rep);c=r['config'];selection=c['select'];reducer=c['reducer']
  c['select']='gain' if selection=='split' else selection
  if reducer=='autoencoder':c['reducer']='none'
  self.pre=Preprocessor(self.bank,r,'linear' if self.model in ['act_logit','act_cloglog','woe_scorecard','cred_scorecard','elastic','spline','rff','tabm','resmlp','ft','tabr','gru','deepsets'] else 'lgb',self.seed)
  # Learned selection must see cross-fitted training encodings, never evaluation labels.
  x=self.pre.fit_transform(rows,y);self.selected=None;self.ae=None
  if reducer=='autoencoder':
   # Denoising autoencoder trained only on the outer-training covariates.
   from sklearn.neural_network import MLPRegressor
   from sklearn.preprocessing import StandardScaler
   self.ae_scale=StandardScaler();z=np.clip(self.ae_scale.fit_transform(x),-5,5);d=min(self.rep['config']['dimensions'],z.shape[1]);ii=np.random.default_rng(self.seed).choice(len(z),min(10000,len(z)),replace=False)
   self.ae=MLPRegressor(hidden_layer_sizes=(d,),activation='tanh',solver='adam',max_iter=20,batch_size=256,early_stopping=True,random_state=self.seed)
   noisy=z[ii]+np.random.default_rng(self.seed).normal(0,.05,z[ii].shape);self.ae.fit(noisy,z[ii]);reduced=np.tanh(z@self.ae.coefs_[0]+self.ae.intercepts_[0]);x=np.column_stack([x,reduced]) if self.rep['config']['append'] else reduced
  self.audit={**self.pre.audit,'actual_selection':selection,'actual_reducer':reducer,'output_features':x.shape[1],'fit_rows':len(rows)}
  return x.astype(np.float32)
 def transform(self,rows):
  x=self.pre.transform(rows)
  if self.selected is not None:x=x[:,self.selected]
  if self.ae is not None:
   z=np.clip(self.ae_scale.transform(x),-5,5);reduced=np.tanh(z@self.ae.coefs_[0]+self.ae.intercepts_[0]);x=np.column_stack([x,reduced]) if self.rep['config']['append'] else reduced
  return x.astype(np.float32)
 def detach(self):self.bank=None;self.pre.bank=None
