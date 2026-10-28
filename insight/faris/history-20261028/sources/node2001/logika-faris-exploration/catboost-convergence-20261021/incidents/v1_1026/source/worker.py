import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json,time,argparse,hashlib,shutil
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for v in iter(lambda:f.read(8388608),b''):h.update(v)
 return h.hexdigest()
def write(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def load(f):
 d=R/f'data/fold{f}';m=read(d/'manifest.json')
 for n,h in m['files'].items():assert sha(d/n)==h
 schema=read(d/'schema.json');tr=pd.read_parquet(d/'train.parquet');va=pd.read_parquet(d/'validation.parquet');ix=dict(np.load(d/'indices.npz'))
 assert len(tr)==56000 and len(va)==14000 and'target'not in va and set(tr.application_id).isdisjoint(va.application_id)
 assert not{'target','split','application_id','customer_id','decision_datetime'}&set(schema['columns'])
 assert np.array_equal(np.sort(np.r_[ix['train'],ix['validation']]),np.arange(70000))
 return tr,va,ix,schema,m

def model(n,f,folder):
 p=read(R/'plan.json')['model'].copy();p.update(iterations=n,devices='0',random_seed=2026+f,train_dir=str(folder),allow_writing_files=True,metric_period=1)
 return CatBoostClassifier(**p)
def guard():
 assert time.time()<read(R/'plan.json')['original_deadline']and not(R/'STOP').exists()
 assert shutil.disk_usage(R).free>2*1024**3

def preflight():
 guard();tr,va,ix,s,meta=load(0);d=R/'preflight';assert not d.exists();d.mkdir();x=tr[s['columns']];y=tr.target.to_numpy(int);cats=s['categorical']
 m=model(20,0,d/'logs');m.fit(Pool(x.iloc[:512],y[:512],cat_features=cats),eval_set=Pool(x.iloc[50000:50128],y[50000:50128],cat_features=cats),use_best_model=False,verbose=False)
 p=m.predict_proba(x.iloc[50000:50128],thread_count=4)[:,1];m.save_model(str(d/'model.cbm'));re=CatBoostClassifier();re.load_model(str(d/'model.cbm'));q=re.predict_proba(x.iloc[50000:50128],thread_count=4)[:,1]
 assert np.max(abs(p-q))<1e-14 and m.get_all_params()['task_type']=='GPU'
 write(R/'PREFLIGHT.json',dict(status='passed',model_task_type='GPU',checkpoint_replay_error=float(np.max(abs(p-q))),outer_labels_available=False,feature_count=len(s['columns']),categorical=cats,model_sha256=sha(d/'model.cbm'),training_context_rows=512,engineering_eval_rows_from_reserved_outer_training=128))

def fold(f):
 guard();assert read(R/'plan.json')['training_admitted']and read(R/'PREFLIGHT.json')['status']=='passed'
 tr,va,ix,s,meta=load(f);d=R/f'folds/fold{f}';assert not d.exists();d.mkdir(parents=True);start=time.time();x=tr[s['columns']];y=tr.target.to_numpy(int);cats=s['categorical'];plan=read(R/'plan.json')
 print(json.dumps(dict(stage='inner_GPU_training',fold=f,context_rows=50000,selection_rows=6000,features=len(s['columns']),tree_cap=5000)),flush=True)
 inner=model(5000,f,d/'inner_logs');inner.fit(Pool(x.iloc[:50000],y[:50000],cat_features=cats),eval_set=Pool(x.iloc[50000:],y[50000:],cat_features=cats),use_best_model=False,early_stopping_rounds=plan['early_stopping_rounds'],verbose=100)
 curve=inner.get_evals_result()['validation']['AUC'];best=int(inner.get_best_iteration())+1;capped=int(np.argmax(curve[:300]))+1;assert 1<=capped<=best<=5000;inner.save_model(str(d/'inner_full.cbm'));write(d/'selected_iteration.json',dict(selected_trees=best,matched_300_cap_trees=capped,inner_retained_trees=inner.tree_count_,selected_on='Reserved6000 only, AUC',outer_labels_loaded_by_worker=False));del inner
 guard();print(json.dumps(dict(stage='full_outer_training_refit',fold=f,rows=56000,selected_trees=best)),flush=True)
 full=model(best,f,d/'full_logs');full.fit(Pool(x,y,cat_features=cats),verbose=100);full.save_model(str(d/'model.cbm'));p=full.predict_proba(va[s['columns']],thread_count=4)[:,1]
 re=CatBoostClassifier();re.load_model(str(d/'model.cbm'));q=re.predict_proba(va[s['columns']],thread_count=4)[:,1];prefix=full.copy();prefix.shrink(ntree_end=capped);prefix.save_model(str(d/'matched_300_prefix.cbm'));cp=prefix.predict_proba(va[s['columns']],thread_count=4)[:,1];assert np.max(abs(cp-full.predict_proba(va[s['columns']],ntree_end=capped,thread_count=4)[:,1]))<1e-14;err=float(np.max(abs(p-q)));assert err<1e-14 and np.isfinite(p).all()
 np.savez_compressed(d/'validation.npz',rows=ix['validation'],application_id=va.application_id.to_numpy(str),prediction=p,matched_300_prediction=cp)
 write(d/'boundary.json',dict(context_rows=50000,selection_rows=6000,refit_rows=56000,outer_rows=14000,outer_labels_loaded_by_worker=False,selected_trees=best,all_model_parameters=full.get_all_params(),feature_columns=s['columns'],categorical=cats,input_manifest=meta,seconds=time.time()-start,replay_error=err,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
 write(d/'complete.json',dict(status='complete',fold=f,files={str(p.relative_to(d)):sha(p)for p in d.rglob('*')if p.is_file()}));print(json.dumps(dict(stage='complete',fold=f,seconds=time.time()-start)),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['preflight','fold']);a.add_argument('--fold',type=int);v=a.parse_args();preflight()if v.action=='preflight'else fold(v.fold)
