import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
os.environ['KUMO_MEMORY_FRACTION']='.30'
from pathlib import Path
import json,time,argparse,gc,hashlib,subprocess
import numpy as np,pandas as pd,joblib
import baseline as b,feature_head
from gpu_backend import Engine
R=Path(__file__).resolve().parents[1];DATA=R.parent/'kumo-temperature-adaptation-20261021/full_data'
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'plan.json')['original_deadline']and not(R/'STOP').exists()
def setup():
 check();p=read(R/'plan.json');assert b.sha(p['checkpoint'])==p['checkpoint_sha256']
 for n,h in p['source_sha256'].items():assert b.sha(R/'source'/n)==h
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert free>=p['minimum_free_VRAM_MiB'],free
 return p
def load(f):
 d=DATA/f'fold{f}';m=read(d/'manifest.json')
 for n,h in m['files'].items():assert b.sha(d/n)==h
 c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');y=np.load(d/'correction_labels.npy').astype(np.int64);ix=dict(np.load(d/'indices.npz'));sp=dict(np.load(d/'spline_validation.npz'));cols=read(R/'columns.json');hcols=read(DATA/'head_columns.json')
 assert len(c)==50000 and len(q)==20000 and len(y)==6000 and len(cols)==90 and len(hcols)==18 and'target'not in q
 assert set(c.application_id).isdisjoint(q.application_id)and c.application_id.is_unique and q.application_id.is_unique
 assert len(set(np.r_[ix['context'],ix['correction'],ix['validation']]))==70000 and np.array_equal(sp['rows'],ix['validation'])
 return c,q,y,ix,sp,cols,hcols,m
def preflight():
 p=setup();assert p['preflight_admitted'];seen=np.zeros(70000,int)
 for f in range(5):c,q,y,ix,*_=load(f);seen[ix['validation']]+=1
 assert np.all(seen==1)
 c,q,y,ix,sp,cols,hcols,m=load(0);d=R/'preflight';assert not d.exists();d.mkdir();query=q.iloc[:128];e=Engine(p['checkpoint'],cols,c.iloc[:512]);raw=e.fit_predict(np.arange(512),query,2026,d/'fitted_context.pt');spectra=e.last_quantiles.copy();assert spectra.shape==(128,999)and np.isfinite(spectra).all();assert np.all(np.diff(spectra,axis=1)>=-1e-5)
 # CPU direct quadrature independently checks the decoder, including boundaries.
 expected=np.clip(np.clip(spectra.astype(np.float64),0,1).mean(axis=1),1e-6,1-1e-6);assert np.max(abs(raw-expected))<2e-7
 assert b.sha(d/'fitted_context.pt')==e.last_cache['sha256'];rp=e.replay(d/'fitted_context.pt',query);assert np.max(abs(raw-rp))<2e-5
 head=feature_head.fit(query.assign(target=y[:128]),raw,hcols,.003,.5);joblib.dump(head,d/'head.joblib');pr=b.corrected_probability(head,query,raw)[0];rr=b.corrected_probability(joblib.load(d/'head.joblib'),query,raw)[0];assert max(abs(pr-rr))<1e-14
 np.savez_compressed(d/'predictions.npz',prediction=raw,replay=rp,quantiles=spectra)
 b.dump(dict(status='passed',actual_GPU=e.torch.cuda.get_device_name(),fivefold_roles_checked=True,query_labels_not_passed_to_backbone=True,outer_labels_staged=False,numeric_target_unchanged=True,quantile_count=999,decoder_independent_max_error=float(np.max(abs(raw-expected))),fitted_cache_replay_max_error=float(np.max(abs(raw-rp))),head_replay_error=float(max(abs(pr-rr)))),R/'PREFLIGHT.json')
def fold(f):
 p=setup();assert p['training_admitted']and read(R/'PREFLIGHT.json')['status']=='passed';c,q,y,ix,sp,cols,hcols,m=load(f);d=R/f'folds/fold{f}';assert not d.exists();d.mkdir(parents=True);start=time.time()
 print(json.dumps(dict(stage='regression_prior_fit',fold=f,context=50000,query=20000,features=90,quantiles=999)),flush=True)
 e=Engine(p['checkpoint'],cols,c);raw=e.fit_predict(np.arange(50000),q,2026+f,d/'fitted_context.pt');spectra=e.last_quantiles;assert spectra.shape==(20000,999)and np.isfinite(spectra).all();np.save(d/'raw_quantiles.npy',spectra);clipfraction=float(np.mean((spectra<0)|(spectra>1)));cache=e.last_cache.copy();assert b.sha(d/'fitted_context.pt')==cache['sha256'];del spectra;rp=e.replay(d/'fitted_context.pt',q);err=float(max(abs(raw-rp)));assert err<2e-5;del e;gc.collect()
 head=feature_head.fit(q.iloc[:6000].assign(target=y),raw[:6000],hcols,.003,.5);joblib.dump(head,d/'correction.joblib',compress=3);pr=.75*b.corrected_probability(head,q.iloc[6000:],raw[6000:])[0]+.25*sp['prediction'];rr=.75*b.corrected_probability(joblib.load(d/'correction.joblib'),q.iloc[6000:],raw[6000:])[0]+.25*sp['prediction'];assert max(abs(pr-rr))<1e-14
 np.savez_compressed(d/'raw.npz',rows=np.r_[ix['correction'],ix['validation']],prediction=raw,replayed_prediction=rp);np.savez_compressed(d/'validation.npz',rows=ix['validation'],application_id=q.application_id.iloc[6000:].to_numpy(str),prediction=pr)
 b.dump(dict(**cache,full_replay_max_error=err),d/'cache.json');b.dump(dict(fold=f,input_manifest=m,numeric_target='binary0/1 unchanged',decoder=p['decoder'],quantiles_outside_binary_bounds_fraction=clipfraction,outer_labels_opened=False,context_rows=50000,head_rows=6000,outer_rows=14000,seconds=time.time()-start),d/'boundary.json');b.dump(dict(status='complete',fold=f,files={z.name:b.sha(z)for z in d.iterdir()if z.is_file()and z.name!='fitted_context.pt'},retained_cache_sha256=cache['sha256']),d/'complete.json');print(json.dumps(dict(stage='complete',fold=f,seconds=time.time()-start)),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['preflight','fold']);a.add_argument('--fold',type=int);v=a.parse_args();preflight()if v.action=='preflight'else fold(v.fold)
