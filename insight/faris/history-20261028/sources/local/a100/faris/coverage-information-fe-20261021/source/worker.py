import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,gc,argparse
import numpy as np,pandas as pd,joblib
import baseline as b,feature_head
from context_model import freeze,predict
from gpu_backend import Engine
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def main(f):
 p=read(R/'training-plan.json');assert p['training_admitted']and time.time()<p['original_deadline']and not(R/'STOP').exists()
 for n,h in p['worker_source_sha256'].items():assert b.sha(R/'source'/n)==h
 checkpoint=Path(p['checkpoint']);assert b.sha(checkpoint)==p['checkpoint_sha256'];d=R/f'data/fold{f}';m=read(d/'manifest.json')
 for n,h in m['files'].items():assert b.sha(d/n)==h
 c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');ix=np.load(d/'indices.npz');cols=read(R/'columns.json');assert len(cols)==90 and len(c)==50000 and len(q)==20000 and'target'not in q
 assert not{'application_id','customer_id','target','split'}&set(cols)and set(c.application_id).isdisjoint(q.application_id)
 root=R/f'runs/fold{f}';assert not root.exists();root.mkdir(parents=True);start=time.time();model=root/'context_model.joblib';seed=2026+f
 freeze(c,cols,seed,checkpoint,model);engine=Engine(checkpoint,cols,c);assert all(not x.requires_grad for x in engine.model.parameters());raw=engine.fit_predict(np.arange(50000),q,seed);del engine;gc.collect()
 repeated=predict(model,checkpoint,q);error=float(abs(raw-repeated).max());assert error<2e-5 and np.isfinite(raw).all()
 # Backbone prediction is complete before the6000 head labels are opened.
 labels=np.load(d/'correction_labels.npy').astype(np.int64);assert len(labels)==6000
 old=joblib.load(R.parent/f'kumo-coverage-placement-20261021/conditions/fold{f}/control90/correction.joblib')
 head=feature_head.fit(q.iloc[:6000].assign(target=labels),raw[:6000],old['columns'],.003,.5);assert len(head['coefficients'])==len(old['coefficients']);joblib.dump(head,root/'head.joblib',compress=3)
 sp=np.load(d/'spline_validation.npz');assert np.array_equal(sp['rows'],ix['validation'])
 pr=.75*b.corrected_probability(head,q.iloc[6000:],raw[6000:])[0]+.25*sp['prediction'];replay=.75*b.corrected_probability(joblib.load(root/'head.joblib'),q.iloc[6000:],repeated[6000:])[0]+.25*sp['prediction']
 assert np.max(abs(pr-replay))<2e-5
 np.savez_compressed(root/'raw.npz',rows=np.r_[ix['correction'],ix['validation']],prediction=raw,replayed_prediction=repeated,application_id=q.application_id.to_numpy(str));np.savez_compressed(root/'validation.npz',rows=ix['validation'],prediction=pr,replayed_prediction=replay,application_id=q.application_id.iloc[6000:].to_numpy(str))
 b.dump(dict(fold=f,context_rows=50000,head_rows=6000,outer_rows=14000,seed=seed,core_columns=cols,all_backbone_parameters_frozen=True,correction_parameters=len(head['coefficients']),feature_boundary=read(d/'boundary.json'),outer_validation_labels_opened=False,query_labels_passed_to_backbone=False,canonical_context_retained=True,attention_cache_checkpoints_created=0,context_cold_replay_max_error=error,final_model_replay_max_error=float(abs(pr-replay).max()),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),seconds=time.time()-start),root/'boundary.json')
 b.dump(dict(status='complete',fold=f,files={p.name:b.sha(p)for p in root.iterdir()if p.is_file()}),root/'complete.json');print(json.dumps(dict(status='complete',fold=f,seconds=time.time()-start)),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--fold',type=int,required=True);main(a.parse_args().fold)
