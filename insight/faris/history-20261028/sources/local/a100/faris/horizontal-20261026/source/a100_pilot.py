"""Compact A100 capacity/replay pilot. No changes to the main experiment."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['SDM_CHUNK_MEMORY_FRACTION']='0.02'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
from pathlib import Path
import argparse,hashlib,importlib,json,time,resource
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch,sdm

p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.packet);out=Path(a.out);out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((root/'MANIFEST.json').read_text())
for n,v in m['files'].items():assert sha(root/n)==v
assert sha(a.checkpoint)=='1ff91f484e19021aaf7d9eff6cc07e6d95a60b4aaad473e83e2ab2c3bfd2b617'
start=time.time();frame=pd.read_parquet(root/'frame.parquet');roles=np.load(root/'roles.npz')
assert list(frame)==m['columns'] and len(frame)==70000 and len(frame.columns)<=100
module=importlib.import_module(sdm.models.KumoTabular.__module__)
assert sha(module.__file__)=='1bdc0ebb4ed74d4025cd64412d6235bde236fb9e49e890e5fa8830c0a9b0e281'
torch.set_num_threads(2);torch.set_grad_enabled(False);torch.cuda.set_per_process_memory_fraction(.70)
def resolve(**kw):
    assert kw['repo_id']=='nvidia/Kumo-Tabular' and kw['filename']=='small/classifier.pt' and kw['revision']=='v1.0.1'
    return a.checkpoint
with patch.object(module,'download_checkpoint',resolve):
    model=sdm.models.KumoTabular(task='classification',size='small',device='cuda')
types={c:'numerical' if pd.api.types.is_numeric_dtype(frame[c]) else 'categorical' for c in frame}
table=sdm.TableTensor.from_pandas(frame,stypes=types,device='cuda')
target=sdm.TableTensor.from_pandas(pd.DataFrame({'target':roles['y_context'].astype(int)}),stypes={'target':'categorical'},device='cuda')
predictions=[]
with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
    model.fit(table[:50000],target,num_estimators=2,estimator_batch_size=1,
              generator=torch.Generator(device='cuda').manual_seed(2026+m['fold']))
    for offset in range(50000,70000,512):
        predictions.append(model.predict(table[offset:offset+512])[:,'1'].numerical.reshape(-1).float().cpu().numpy())
    repeated=model.predict(table[50000:50512])[:,'1'].numerical.reshape(-1).float().cpu().numpy()
raw=np.concatenate(predictions);error=float(np.max(abs(repeated-raw[:512])))
assert error<1e-7 and len(raw)==20000 and np.isfinite(raw).all()
np.savez_compressed(out/'prediction.npz',rows=roles['rows'][50000:],raw=raw)
(out/'RESULT.json').write_text(json.dumps({'status':'complete','seconds':time.time()-start,
 'pack':m['pack'],'fold':m['fold'],'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
 'replay_max_abs':error,'GPU':torch.cuda.get_device_name(0),'source_sha256':sha(__file__),
 'prediction_sha256':sha(out/'prediction.npz'),'fitted_weights_saved':False,
 'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__},indent=2)+'\n')
print((out/'RESULT.json').read_text(),flush=True)
