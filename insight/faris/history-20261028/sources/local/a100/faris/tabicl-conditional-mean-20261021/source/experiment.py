"""One fixed compact90 TabICLv2 recipe; no per-fold performance pruning."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time,argparse,gc,hashlib,shutil
import numpy as np
import pandas as pd
import joblib,torch
from tabicl import TabICLRegressor
import baseline as b
import feature_head
R=Path(__file__).resolve().parents[1]
STAGED=R.parent/'kumo-temperature-adaptation-20261021/full_data'
MAIN=R.parent/'kumo-coverage-placement-20261021'
DEADLINE=1792585871.823748

def read(p):return json.loads(Path(p).read_text())
def write(p,x):b.dump(x,p)
def frame(x,cols):
 x=x[cols].copy()
 for c in cols:
  if pd.api.types.is_numeric_dtype(x[c]):x[c]=x[c].replace([np.inf,-np.inf],np.nan)
  else:x[c]=x[c].fillna('MISSING').astype(str)
 return x

def data(f):
 d=STAGED/f'fold{f}';m=read(d/'manifest.json')
 for n,h in m['files'].items():assert b.sha(d/n)==h
 c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');ix=dict(np.load(d/'indices.npz'))
 cols=read(R/'CORE_COLUMNS.json')
 assert len(cols)==90 and 'target'not in cols and set(c.application_id).isdisjoint(q.application_id)
 return c,q,ix,cols,read(STAGED/'head_columns.json'),np.load(d/'correction_labels.npy'),dict(np.load(d/'spline_validation.npz'))

def model(seed,folder):
 w=read(R/'WEIGHT.json');assert w['sha256']==read(R/'plan.json')['checkpoint_sha256']
 return TabICLRegressor(n_estimators=8,model_path=w['path'],allow_auto_download=False,checkpoint_version='tabicl-regressor-v2-20260212.ckpt',device='cuda',
  n_jobs=2,batch_size=1,random_state=seed,kv_cache=True,offload_mode='auto',disk_offload_dir=str(folder/'offload'))

UNCLIPPED=[]
def probability(m,x):
 raw=np.asarray(m.predict(x)).reshape(-1);assert np.isfinite(raw).all();UNCLIPPED.append(raw.copy());return np.clip(raw,1e-6,1-1e-6)

def setup():
 assert time.time()<DEADLINE and not(R/'STOP').exists()
 torch.set_num_threads(2);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.45)
 assert shutil.disk_usage(R).free>100*1024**3
 import subprocess
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert free>40000,f'Insufficient safe free VRAM:{free}'

def preflight():
 setup();d=R/'preflight';assert not d.exists();d.mkdir();c,q,ix,cols,*_=data(0)
 rng=np.random.default_rng(20261110);a=rng.choice(len(c),544,replace=False);m=model(2026,d);start=time.time()
 m.fit(frame(c.iloc[a[:512]],cols),c.target.iloc[a[:512]].to_numpy());p=probability(m,frame(c.iloc[a[512:]],cols))
 assert np.isfinite(p).all();m.save(str(d/'fitted.pkl'),save_model_weights=False,save_training_data=True,save_kv_cache=True)
 del m;gc.collect();torch.cuda.empty_cache();m=TabICLRegressor.load(str(d/'fitted.pkl'),device='cuda');replay=probability(m,frame(c.iloc[a[512:]],cols))
 err=float(np.max(abs(p-replay)));assert err<2e-6
 write(R/'PREFLIGHT.json',dict(status='passed',query_labels_not_used=True,training_context_only=True,
  rows512_fit32_query=True,retained_model_replay_error=err,peak_VRAM_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,
  device=torch.cuda.get_device_name(),model_sha256=b.sha(d/'fitted.pkl'),checkpoint_sha256=read(R/'WEIGHT.json')['sha256']))

def fold(f):
 setup();assert read(R/'PREFLIGHT.json')['status']=='passed';assert read(R/'plan.json')['training_admitted']is True
 d=R/f'folds/fold{f}';assert not d.exists(),'Preserve existing attempt';d.mkdir(parents=True);start=time.time()
 c,q,ix,cols,hcols,labels,sp=data(f);assert len(c)==50000 and len(q)==20000 and len(labels)==6000
 assert np.array_equal(sp['rows'],ix['validation']);m=model(2026+f,d)
 print(json.dumps(dict(stage='fit',fold=f,context_rows=50000,features=90,seed=2026+f)),flush=True)
 m.fit(frame(c,cols),c.target.to_numpy());raw=[]
 for i in range(0,len(q),512):
  assert time.time()<DEADLINE and not(R/'STOP').exists()
  raw.append(probability(m,frame(q.iloc[i:i+512],cols)));print(json.dumps(dict(stage='predict',fold=f,rows=min(i+512,len(q)))),flush=True)
 raw=np.concatenate(raw);assert np.isfinite(raw).all()
 unclip=np.concatenate(UNCLIPPED);assert len(unclip)==20000;np.savez_compressed(d/'unclipped.npz',rows=np.r_[ix['correction'],ix['validation']],prediction=unclip)
 m.save(str(d/'fitted.pkl'),save_model_weights=False,save_training_data=True,save_kv_cache=True)
 head_frame=q.iloc[:6000].copy();head_frame['target']=labels;head=feature_head.fit(head_frame,raw[:6000],hcols,.003,.5)
 joblib.dump(head,d/'correction.joblib',compress=3);p=.75*b.corrected_probability(head,q.iloc[6000:],raw[6000:])[0]+.25*sp['prediction']
 np.savez_compressed(d/'validation.npz',rows=ix['validation'],prediction=p,application_id=q.application_id.iloc[6000:].to_numpy(str))
 np.savez_compressed(d/'raw.npz',rows=np.r_[ix['correction'],ix['validation']],prediction=raw)
 write(d/'boundary.json',dict(context_label_rows=50000,correction_label_rows=6000,validation_rows=14000,backbone_columns=cols,
  correction_columns=hcols,validation_targets_loaded=False,preprocessing_fit_context_only=True,peak_VRAM_bytes=torch.cuda.max_memory_allocated(),
  seconds=time.time()-start,device=torch.cuda.get_device_name(),regression_clipped_rows=int(np.sum((unclip<1e-6)|(unclip>1-1e-6))),unclipped_min=float(unclip.min()),unclipped_max=float(unclip.max())))
 # Full fitted-model replay for every query; no performance metric until all folds.
 del m;gc.collect();torch.cuda.empty_cache();m=TabICLRegressor.load(str(d/'fitted.pkl'),device='cuda')
 err=0.
 for i in range(0,len(q),512):err=max(err,float(np.max(abs(probability(m,frame(q.iloc[i:i+512],cols))-raw[i:i+512]))))
 assert err<2e-6
 write(d/'complete.json',dict(status='complete',fold=f,full_replay_max_error=err,
  files={str(x.relative_to(d)):b.sha(x)for x in d.rglob('*')if x.is_file()and x.name!='complete.json'}))
 print(json.dumps(dict(stage='complete',fold=f,seconds=time.time()-start,replay_error=err)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','fold']);p.add_argument('--fold',type=int);a=p.parse_args()
 preflight()if a.action=='preflight'else fold(a.fold)
