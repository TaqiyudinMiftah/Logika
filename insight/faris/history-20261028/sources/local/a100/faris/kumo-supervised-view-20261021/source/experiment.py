"""One supervised feature-view recipe, complete honest fivefold OOF."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,gc,argparse
import numpy as np,pandas as pd,joblib
import baseline as b,feature_head
from gpu_backend import Engine
R=Path(__file__).resolve().parents[1];SHARED=R.parent/'kumo-temperature-adaptation-20261021/full_data'
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'plan.json')['original_deadline']and not(R/'STOP').exists()
def load(f):
 d=R/f'data/fold{f}';m=read(d/'manifest.json');assert m['status']=='complete'and m['feature_selection_rows']==50000 and m['outer_labels_used']is False
 for n,h in m['files'].items():assert b.sha(d/n)==h
 assert b.sha(d/'selector.cbm')==m['selector_model_sha256']
 c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');y=np.load(d/'correction_labels.npy');ix=dict(np.load(d/'indices.npz'));cols=read(d/'columns.json');selection=read(d/'selection.json')
 assert len(c)==50000 and len(q)==20000 and len(y)==6000 and len(cols)==90 and len(set(cols))==90 and'target'not in q
 assert selection['reserved6000_used_in_selection']is False and selection['outer_labels_available']is False
 assert c.application_id.is_unique and q.application_id.is_unique and set(c.application_id).isdisjoint(q.application_id)
 assert len(set(np.r_[ix['context'],ix['correction'],ix['validation']]))==70000
 old=SHARED/f'fold{f}';om=read(old/'manifest.json')
 for n,h in om['files'].items():assert b.sha(old/n)==h
 original_c=pd.read_parquet(old/'context.parquet',columns=['application_id','target']);original_q=pd.read_parquet(old/'query.parquet',columns=['application_id'])
 assert np.array_equal(c.application_id,original_c.application_id)and np.array_equal(c.target,original_c.target)and np.array_equal(q.application_id,original_q.application_id)
 assert np.array_equal(y,np.load(old/'correction_labels.npy'))
 for k,v in dict(np.load(old/'indices.npz')).items():assert np.array_equal(ix[k],v)
 sp=dict(np.load(old/'spline_validation.npz'));assert np.array_equal(sp['rows'],ix['validation'])
 return c,q,y,ix,cols,sp,m

def preflight():
 check();seen=np.zeros(70000,int)
 for f in range(5):
  c,q,y,ix,cols,sp,m=load(f);seen[ix['validation']]+=1
 assert np.all(seen==1)
 c,q,y,ix,cols,sp,m=load(0);d=R/'preflight';assert not d.exists();d.mkdir()
 e=Engine(read(R/'plan.json')['checkpoint'],cols,c.iloc[:512]);p=e.fit_predict(np.arange(512),q.iloc[:128],2026,d/'fitted_context.pt');re=e.replay(d/'fitted_context.pt',q.iloc[:128]);assert np.max(abs(p-re))<2e-5 and b.sha(d/'fitted_context.pt')==e.last_cache['sha256']
 h=feature_head.fit(q.iloc[:128].assign(target=y[:128]),p,read(R/'head_columns.json'),.003,.5);joblib.dump(h,d/'head.joblib');pp=b.corrected_probability(h,q.iloc[:128],p)[0];rr=b.corrected_probability(joblib.load(d/'head.joblib'),q.iloc[:128],p)[0];assert np.max(abs(pp-rr))<1e-14
 b.dump(dict(status='passed',all5_source_and_role_checks=True,GPU=e.torch.cuda.get_device_name(),full_cache_replay_error=float(np.max(abs(p-re))),head_replay_error=float(np.max(abs(pp-rr))),outer_labels_opened=False),R/'PREFLIGHT.json')
def fold(f):
 check();p=read(R/'plan.json');assert p['model_training_admitted']and read(R/'PREFLIGHT.json')['status']=='passed'
 for n,h in p['source_sha256'].items():assert b.sha(R/'source'/n)==h
 assert b.sha(p['checkpoint'])==p['checkpoint_sha256']
 c,q,y,ix,cols,sp,m=load(f);d=R/f'folds/fold{f}';assert not d.exists();d.mkdir(parents=True);start=time.time()
 print(json.dumps(dict(stage='Kumo_context_fit',fold=f,features=90,context=50000,query=20000)),flush=True)
 e=Engine(p['checkpoint'],cols,c);raw=e.fit_predict(np.arange(50000),q,2026+f,d/'fitted_context.pt');cache=e.last_cache.copy();assert b.sha(d/'fitted_context.pt')==cache['sha256'];re=e.replay(d/'fitted_context.pt',q);err=float(np.max(abs(raw-re)));assert err<2e-5
 del e;gc.collect();head=feature_head.fit(q.iloc[:6000].assign(target=y),raw[:6000],read(R/'head_columns.json'),.003,.5);joblib.dump(head,d/'correction.joblib',compress=3);pr=.75*b.corrected_probability(head,q.iloc[6000:],raw[6000:])[0]+.25*sp['prediction'];rr=.75*b.corrected_probability(joblib.load(d/'correction.joblib'),q.iloc[6000:],raw[6000:])[0]+.25*sp['prediction'];assert np.max(abs(pr-rr))<1e-14
 np.savez_compressed(d/'raw.npz',rows=np.r_[ix['correction'],ix['validation']],prediction=raw);np.savez_compressed(d/'validation.npz',rows=ix['validation'],application_id=q.application_id.iloc[6000:].to_numpy(str),prediction=pr)
 b.dump(dict(**cache,full_replay_max_error=err),d/'cache.json');b.dump(dict(feature_selection_rows=50000,head_fit_rows=6000,outer_rows=14000,outer_labels_opened=False,manifest=m,seconds=time.time()-start),d/'boundary.json')
 b.dump(dict(status='complete',fold=f,files={z.name:b.sha(z)for z in d.iterdir()if z.is_file()and z.name!='fitted_context.pt'},retained_cache_sha256=cache['sha256']),d/'complete.json')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['preflight','fold']);a.add_argument('--fold',type=int);v=a.parse_args();preflight()if v.action=='preflight'else fold(v.fold)
