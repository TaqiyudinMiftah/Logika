"""One context-size contrast, retaining an honest heldout correction."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['KUMO_MEMORY_FRACTION']='.30'
from pathlib import Path
import json,time,gc,argparse,hashlib
import numpy as np,pandas as pd,joblib
import baseline as b
from gpu_backend import Engine
R=Path(__file__).resolve().parents[1];DATA=R.parent/'kumo-temperature-adaptation-20261021/full_data'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):b.dump(x,p)
def check():assert time.time()<read(R/'plan.json')['original_deadline']and not(R/'STOP').exists()
def load(f):
 d=DATA/f'fold{f}';m=read(d/'manifest.json')
 for n,h in m['files'].items():assert b.sha(d/n)==h
 c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');y=np.load(d/'correction_labels.npy');ix=dict(np.load(d/'indices.npz'));sp=dict(np.load(d/'spline_validation.npz'))
 assert len(c)==50000 and len(q)==20000 and len(y)==6000 and'target'not in q
 assert set(c.application_id).isdisjoint(q.application_id)and c.application_id.is_unique and q.application_id.is_unique
 assert len(set(np.r_[ix['context'],ix['correction'],ix['validation']]))==70000
 assert len(ix['context'])==50000 and len(ix['correction'])==6000 and len(ix['validation'])==14000
 assert np.array_equal(ix['validation'],sp['rows'])
 h=R/f'heads/control90_head_fold{f}.joblib';assert b.sha(h)==read(R/'head_manifest.json')[h.name]
 head=joblib.load(h);assert len(head['columns'])==18 and head['spec']=={'correction_strength':.5,'penalty':.003}
 return c,q,y,ix,sp,head,m

def preflight():
 assert read(R/'plan.json')['preflight_admitted'];check();seen=np.zeros(70000,int)
 for f in range(5):
  c,q,y,ix,sp,h,m=load(f);seen[ix['validation']]+=1
 assert np.all(seen==1)
 c,q,y,ix,sp,h,m=load(0);small=c.iloc[:512].copy();query=q.iloc[6000:6064].copy();full=pd.concat([small,q.iloc[:64].assign(target=y[:64])],ignore_index=True)
 assert set(full.application_id).isdisjoint(query.application_id)
 d=R/'preflight';assert not d.exists();d.mkdir()
 eng=Engine(read(R/'plan.json')['checkpoint'],read(R/'columns.json'),full);p=eng.fit_predict(np.arange(576),query,2026,d/'fitted_context.pt');re=eng.replay(d/'fitted_context.pt',query);assert np.max(abs(p-re))<2e-5
 assert b.sha(d/'fitted_context.pt')==eng.last_cache['sha256'];write(R/'PREFLIGHT.json',dict(status='passed',fivefold_roles_checked=True,cache_replay_error=float(np.max(abs(p-re))),actual_GPU=eng.torch.cuda.get_device_name(),added_calibration_targets_only_in_final_context=True,outer_label_input_available=False))

def fold(f):
 check();plan=read(R/'plan.json');assert plan['training_admitted']and read(R/'PREFLIGHT.json')['status']=='passed'
 for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
 assert b.sha(plan['checkpoint'])==plan['checkpoint_sha256']
 import subprocess
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert free>=plan['minimum_free_VRAM_MiB']
 c,q,y,ix,sp,head,m=load(f);v=q.iloc[6000:].copy();d=R/f'folds/fold{f}';assert not d.exists();d.mkdir(parents=True);start=time.time();pred={}
 for size in [50000,56000]:
  check();context=c if size==50000 else pd.concat([c,q.iloc[:6000].assign(target=y)],ignore_index=True)
  assert len(context)==size and set(context.application_id).isdisjoint(v.application_id)
  assert len(set(context.application_id))==size
  arm=d/f'context{size}';arm.mkdir();print(json.dumps(dict(stage='fit',fold=f,context=size,query_rows=14000)),flush=True)
  engine=Engine(plan['checkpoint'],read(R/'columns.json'),context);p=engine.fit_predict(np.arange(size),v,2026+f,arm/'fitted_context.pt');cache=engine.last_cache.copy();assert b.sha(arm/'fitted_context.pt')==cache['sha256']
  # Replay every outer prediction from the retained fitted cache.
  re=engine.replay(arm/'fitted_context.pt',v);err=float(np.max(abs(p-re)));assert err<2e-5
  pred[str(size)]=.75*b.corrected_probability(head,v,p)[0]+.25*sp['prediction']
  np.savez_compressed(arm/'raw.npz',rows=ix['validation'],application_id=v.application_id.to_numpy(str),prediction=p)
  write(arm/'cache.json',dict(**cache,full_replay_max_error=err));del engine;gc.collect()
 np.savez_compressed(d/'validation.npz',rows=ix['validation'],application_id=v.application_id.to_numpy(str),context50000=pred['50000'],context56000=pred['56000'])
 write(d/'boundary.json',dict(fold=f,input_manifest=m,correction_labels_reused_only_as_outer_training_labels=True,head_refitted=False,outer_label_input_available=False,seconds=time.time()-start))
 write(d/'complete.json',dict(status='complete',fold=f,files={str(p.relative_to(d)):b.sha(p)for p in d.rglob('*')if p.is_file()and p.name!='fitted_context.pt'},retained_cache_hashes={str(n):read(d/f'context{n}/cache.json')['sha256']for n in [50000,56000]}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['preflight','fold']);a.add_argument('--fold',type=int);v=a.parse_args();preflight()if v.action=='preflight'else fold(v.fold)
