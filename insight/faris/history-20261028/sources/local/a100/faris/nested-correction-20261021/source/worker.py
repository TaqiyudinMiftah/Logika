"""Raw frozen ICL predictions; query labels are absent from GPU packets."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
from pathlib import Path
import json,time,argparse,gc,numpy as np,pandas as pd
from gpu_backend import Engine
from context_model import freeze,predict,sha
R=Path(__file__).resolve().parents[1]
def run(j):
 p=json.loads((R/'training-plan.json').read_text());assert p['training_admitted']and time.time()<p['original_deadline']and not(R/'STOP').exists()
 for n,h in p['worker_source_sha256'].items():assert sha(R/'source'/n)==h
 checkpoint=Path(os.environ['CHECKPOINT_PATH']);assert sha(checkpoint)==p['checkpoint_sha256']
 d=R/f'data/job{j}';m=json.loads((d/'manifest.json').read_text());assert m['job']==j and not m['query_labels_staged']
 for n,h in m['files'].items():assert sha(d/n)==h
 c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');ix=dict(np.load(d/'indices.npz'));cols=json.loads((R/'columns.json').read_text());job=m['role']
 assert len(c)==job['context_rows']and len(q)==job['query_rows']and'target'not in q and len(cols)==90
 assert len(ix['context'])==len(c)and len(ix['validation'])==len(q)and set(ix['context']).isdisjoint(ix['validation'])
 assert set(c.application_id).isdisjoint(q.application_id)and c.application_id.is_unique and q.application_id.is_unique
 import hashlib
 assert hashlib.sha256('\n'.join(c.application_id).encode()).hexdigest()==m['feature_prior_context_ids_sha256']
 out=R/f'runs/job{j}';assert not out.exists(),'Preserve partial fit; explicit owner review required';out.mkdir(parents=True);start=time.time();seed=2026+job['outer_fold']
 model=out/'model.joblib';freeze(c,cols,seed,checkpoint,model);e=Engine(checkpoint,cols,c);assert all(not z.requires_grad for z in e.model.parameters());raw=e.fit_predict(np.arange(len(c)),q,seed);del e;gc.collect()
 # Cold reconstruction verifies the complete retained context model, not an in-memory object.
 replay=predict(model,checkpoint,q);err=float(np.max(abs(raw-replay)));assert err<2e-5 and np.isfinite(raw).all()
 np.savez_compressed(out/'raw.npz',rows=ix['validation'],application_id=q.application_id.to_numpy(str),prediction=raw,replayed_prediction=replay)
 (out/'boundary.json').write_text(json.dumps(dict(job=j,role=job,input_manifest=m,query_labels_opened=False,canonical_ICL_model_retained=True,cache_checkpoints_created=0,all_pretrained_parameters_frozen=True,full_cold_model_replay_max_error=err,seconds=time.time()-start,device=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2))
 (out/'complete.json').write_text(json.dumps(dict(status='complete',job=j,files={z.name:sha(z)for z in out.iterdir()if z.is_file()}),indent=2));print(json.dumps(dict(job=j,status='complete',seconds=time.time()-start)),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--job',type=int,required=True);run(a.parse_args().job)
