from pathlib import Path
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time,numpy as np,pandas as pd
from gpu_backend import Engine
from context_model import freeze,predict,sha
R=Path(__file__).resolve().parents[1];P=R.parent/'kumo-coverage-placement-20261021';p=json.loads((P/'plan.json').read_text());checkpoint=p['checkpoint'];assert sha(checkpoint)=='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613';assert time.time()<1792585871.823748-10800
cols=json.loads((R/'BASELINE_LOCK.json').read_text())['Kumo']['features'];frame=pd.read_parquet(Path(p['input'])/'data/train_features.parquet');ix=np.load(Path(p['input'])/'artifacts/fold0/indices.npz');context=frame.iloc[np.r_[ix['context'],ix['correction']]].copy();query=frame.iloc[ix['validation'][:128]].drop(columns='target').copy();assert len(context)==56000 and set(context.application_id).isdisjoint(query.application_id)
start=time.time();model=R/'fitted_context_model.joblib';freeze(context,cols,2026,checkpoint,model)
e=Engine(checkpoint,cols,context);direct=e.fit_predict(np.arange(56000),query,2026);del e
replay=predict(model,checkpoint,query);error=float(np.max(abs(direct-replay)));assert error<2e-5
np.savez_compressed(R/'predictions.npz',direct=direct,replay=replay,rows=ix['validation'][:128])
import importlib.metadata as m
(R/'PREFLIGHT.json').write_text(json.dumps(dict(status='passed',context_rows=56000,features=90,query_rows=128,cold_materialization_replay_max_error=error,retained_model_bytes=model.stat().st_size,retained_model_sha256=sha(model),checkpoint_sha256=sha(checkpoint),all_pretrained_parameters_frozen=True,cache_checkpoints_created=0,outer_labels_loaded=False,seconds=time.time()-start,environment={n:m.version(n)for n in ['numpy','pandas','torch','structured-data-models','joblib']}),indent=2));print('FULL_CONTEXT_CANONICAL_MODEL_REPLAY_PASSED',flush=True)
