"""Small deterministic evidence helpers; no service credentials."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def ident(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ks(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,float)
 assert len(y)==len(p) and set(np.unique(y))=={0,1} and np.isfinite(p).all()
 ix=np.argsort(p,kind='stable');s=y[ix];v=p[ix];ends=np.r_[v[1:]!=v[:-1],True]
 d=np.cumsum(s)/s.sum()-np.cumsum(1-s)/(1-s).sum();return float(np.max(np.abs(d[ends])))
def combine(spec,arrays):
 x=np.column_stack([arrays[k] for k in spec['parents']]);w=np.asarray(spec.get('weights',[1/x.shape[1]]*x.shape[1]));assert np.isfinite(x).all();m=spec['method']
 if m=='trimmed':return np.sort(x,axis=1)[:,1:-1].mean(axis=1)
 if m=='mean':return x@w
 if m=='geometric':return np.exp(np.log(np.clip(x,1e-8,1))@w)
 if m=='logit':
  a=np.clip(x,1e-8,1-1e-8);return 1/(1+np.exp(-(np.log(a/(1-a))@w)))
 if m=='rank':
  from scipy.stats import rankdata
  return np.column_stack([rankdata(x[:,i])/len(x) for i in range(x.shape[1])])@w
 raise ValueError(m)
def delta(y,p,b,fold):
 ds=[ks(y[fold==f],p[fold==f])-ks(y[fold==f],b[fold==f]) for f in range(3)]
 return {'ks':ks(y,p),'delta':ks(y,p)-ks(y,b),'fold_delta':ds,'eligible':bool(sum(d>0 for d in ds)>=2 and min(ds)>=-.005)}
