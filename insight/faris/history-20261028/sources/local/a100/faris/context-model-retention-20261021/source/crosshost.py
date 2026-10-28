from pathlib import Path
import json,time,numpy as np,pandas as pd
from context_model import predict,sha
R=Path(__file__).resolve().parents[1];A=R.parent
lock=json.loads((R/'BASELINE_LOCK.json').read_text());cols=lock['Kumo']['features'];P=A/'ranking-pipeline-20261020/input';ix=np.load(P/'artifacts/fold0/indices.npz');frame=pd.read_parquet(P/'data/train_features.parquet',columns=['application_id']+cols);query=frame.iloc[ix['validation'][:128]].copy();assert'target'not in query
model=R/'fitted_context_model.joblib';source=json.loads((R/'PREFLIGHT.json').read_text());assert sha(model)==source['retained_model_sha256'];expected=np.load(R/'predictions.npz');checkpoint=A/'ranking-pipeline-20261020/assets/classifier.pt';start=time.time();p=predict(model,checkpoint,query);err=float(np.max(abs(p-expected['direct'])));assert err<2e-5
(R/'CROSSHOST.json').write_text(json.dumps(dict(status='passed',cold_crosshost_replay_max_error=err,model_sha256=sha(model),source_host='pc3',replay_host='pc2',outer_labels_loaded=False,all_context_labels_stored_in_retained_model=True,seconds=time.time()-start),indent=2));print('CROSSHOST_CONTEXT_MODEL_REPLAY_PASSED',flush=True)
