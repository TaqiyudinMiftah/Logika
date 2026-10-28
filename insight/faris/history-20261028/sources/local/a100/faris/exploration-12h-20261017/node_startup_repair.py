"""Repair the node's pre-training staging error; retain the original clock."""
from pathlib import Path
import os,time,json,hashlib,shutil,sys
ROOT=Path(__file__).resolve().parent
clock=json.loads((ROOT/'clock.json').read_text());assert time.time()<clock['start']+9*3600
try:
 from huggingface_hub import hf_hub_download
 source=hf_hub_download('jingang/TabICL','tabicl-classifier-v2-20260212.ckpt',token=False)
 dest=ROOT/'weights/tabicl-classifier-v2-20260212.ckpt';shutil.copy2(source,dest)
 (ROOT/'evidence/node-weights.json').write_text(json.dumps({dest.name:hashlib.sha256(dest.read_bytes()).hexdigest()}))
except Exception as e:
 (ROOT/'evidence/tabicl-exclusion.json').write_text(json.dumps({'reason':type(e).__name__,'checkpoint_unavailable':True}))
if (ROOT/'campaign').exists() and not (ROOT/'campaign/seal.json').exists():(ROOT/'campaign').rename(ROOT/('failed-unsealed-bundle-'+str(int(time.time()))))
os.environ.update(HF_HOME='/home/taqiyudinmiftah/faris/kumo-large-node380-20261016/hf_cache',HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLBACKEND='Agg')
os.execv(sys.executable,[sys.executable,str(ROOT/'bootstrap.py'),'node380'])
