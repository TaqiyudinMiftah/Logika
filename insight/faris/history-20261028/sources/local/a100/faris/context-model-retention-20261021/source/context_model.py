"""Canonical frozen-ICL model: immutable context plus shared pretrained weights.

No optimized parameter is discarded. Materializing attention caches is inference
work, analogous to rebuilding a neighbor-search index over retained training data.
Existing cache checkpoints stay preserved. This module creates no cache checkpoint.
"""
from pathlib import Path
import hashlib,joblib,numpy as np
from gpu_backend import Engine

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def freeze(context,columns,seed,checkpoint,output):
 assert context.application_id.is_unique and not{'application_id','customer_id','target','split'}&set(columns)
 assert not Path(output).exists()
 payload=dict(kind='frozen_context_ICL_model_v1',context=context[['application_id','target']+columns].copy(deep=True),columns=list(columns),seed=int(seed),checkpoint_sha256=sha(checkpoint),num_estimators=8,dtype='float16',engine_sha256=sha(Path(__file__).with_name('gpu_backend.py')),model_code_sha256=sha(Path(__file__)))
 joblib.dump(payload,output,compress=3)
def predict(model_path,checkpoint,query):
 p=joblib.load(model_path);assert p['kind']=='frozen_context_ICL_model_v1'and p['checkpoint_sha256']==sha(checkpoint)
 assert p['engine_sha256']==sha(Path(__file__).with_name('gpu_backend.py'))and p['model_code_sha256']==sha(Path(__file__))
 assert set(p['context'].application_id).isdisjoint(query.application_id)and'target'not in query
 engine=Engine(checkpoint,p['columns'],p['context']);assert all(not v.requires_grad for v in engine.model.parameters())
 return engine.fit_predict(np.arange(len(p['context'])),query,p['seed'],cache_path=None)
