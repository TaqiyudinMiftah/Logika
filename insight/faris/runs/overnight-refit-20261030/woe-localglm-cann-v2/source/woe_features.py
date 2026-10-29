"""Cross-fitted smoothed bad-risk enrichment, a signed WoE/credibility adaptation.

Only inner training labels enter maps. OOF assignments are label-independent and
held fixed; own-row labels cannot enter that row's encoded training features.
"""
import numpy as np,pandas as pd
from scipy import sparse
from scipy.special import logit
from sklearn.model_selection import KFold
from cpu_offset import FrozenInput,_hash,_ids
class WoeInput(FrozenInput):
 def __init__(self,mode,bins,strength):self.mode=mode;self.bins=int(bins);self.strength=float(strength);assert mode in {'append','woe_only'}
 def fit_supervised(self,frame,rows,y):
  rows=np.asarray(rows,np.int64);y=np.asarray(y,np.int64);assert len(rows)==len(y) and set(np.unique(y))=={0,1}
  super().fit(frame,rows);self.edges={};self.categories={};self.sizes={}
  for c in self.numeric:
   a=pd.to_numeric(frame[c].iloc[rows],errors='coerce').to_numpy(dtype=float,na_value=np.nan);a=a[np.isfinite(a)]
   self.edges[c]=np.unique(np.quantile(a,np.linspace(0,1,self.bins+1)[1:-1]))if len(a)else np.array([])
   self.sizes[c]=len(self.edges[c])+2
  for c in self.categorical:
   a=frame[c].iloc[rows].astype('string').fillna('__MISSING__');levels=a.value_counts().head(63).index.tolist();self.categories[c]={str(v):i for i,v in enumerate(levels)};self.sizes[c]=len(levels)+1
  codes=self._codes(frame,rows);self.maps=self._fit_maps(codes,y);self.oof=np.zeros(codes.shape);fold=np.empty(len(rows),dtype=np.int64);self.crossfit_row_hashes=[]
  for k,(tr,va) in enumerate(KFold(3,shuffle=True,random_state=20261030).split(rows)):
   maps=self._fit_maps(codes[tr],y[tr]);self.oof[va]=self._lookup(codes[va],maps);fold[va]=k
   self.crossfit_row_hashes.append({'train_rows_sha256':_ids(rows[tr]),'held_rows_sha256':_ids(rows[va])})
  self.positions=np.full(len(frame),-1,dtype=np.int64);self.positions[rows]=np.arange(len(rows));self.fold_hash=_hash(fold);self.target_hash=_hash(y);return self
 def _codes(self,frame,rows):
  out=[]
  for c in self.columns:
   if c in self.edges:
    a=pd.to_numeric(frame[c].iloc[rows],errors='coerce').to_numpy(dtype=float,na_value=np.nan);v=np.searchsorted(self.edges[c],a,side='right');v[~np.isfinite(a)]=self.sizes[c]-1
   else:
    a=frame[c].iloc[rows].astype('string').fillna('__MISSING__');v=a.map(self.categories[c]).fillna(self.sizes[c]-1).to_numpy(dtype=np.int64)
   out.append(v)
  return np.column_stack(out).astype(np.int64)
 def _fit_maps(self,codes,y):
  prior=float(np.clip(np.mean(y),1e-6,1-1e-6));out=[]
  for j,c in enumerate(self.columns):
   n=np.bincount(codes[:,j],minlength=self.sizes[c]);k=np.bincount(codes[:,j],weights=y,minlength=self.sizes[c]);p=(k+self.strength*prior)/(n+self.strength)
   out.append(np.clip(logit(np.clip(p,1e-8,1-1e-8))-logit(prior),-8,8))
  return out
 def _lookup(self,codes,maps):return np.column_stack([m[codes[:,j]]for j,m in enumerate(maps)])
 def glm(self,frame,rows):
  rows=np.asarray(rows,np.int64);z=self._lookup(self._codes(frame,rows),self.maps);pos=self.positions[rows];known=pos>=0;z[known]=self.oof[pos[known]];assert np.isfinite(z).all()
  if self.mode=='append':return sparse.hstack([super().glm(frame,rows),sparse.csr_matrix(z)],format='csr')
  missing=~np.isfinite(self._numeric(frame,rows));return sparse.hstack([sparse.csr_matrix(z),sparse.csr_matrix(missing.astype(float))],format='csr')
 def recipe(self):
  v=super().recipe();v.update(feature_engineering_mode='crossfit_bad_risk_enrichment_'+self.mode,bins=self.bins,beta_binomial_prior_strength=self.strength,folds=3,fold_seed=20261030,label_independent_folds=True,fold_sha256=self.fold_hash,crossfit_rows=self.crossfit_row_hashes,permitted_inner_target_sha256=self.target_hash,full_inner_map_sha256=[_hash(x)for x in self.maps],OOF_feature_sha256=_hash(self.oof),numeric_edges_sha256={c:_hash(v)for c,v in self.edges.items()},own_row_label_excluded=True,retains_inner_encoder_at_full_context_refit=True,no_global_supervised_rescaling=True,fitted_weights_saved=False);return v
