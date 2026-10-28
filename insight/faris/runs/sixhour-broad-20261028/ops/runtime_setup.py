"""Initialize only this campaign's owned runtime; no competition fits here."""
from pathlib import Path
import json
import subprocess
import sys
import threading
import time

R = Path.home() / 'logika-sixhour-broad-20261028'
R.mkdir(exist_ok=True)
# Controller replaces this literal with the immutable clock before execution.
DEADLINE = float('__DEADLINE_EPOCH__')
CHECKPOINTS = json.loads('__CHECKPOINTS_JSON__')
assert time.time() < DEADLINE

def release():
    while time.time() < DEADLINE:
        time.sleep(max(0., min(30., DEADLINE - time.time())))
    from google.colab import runtime
    runtime.unassign()

if not (R / 'RELEASE_WATCHDOG.json').exists():
    threading.Thread(target=release, daemon=True).start()
    (R / 'RELEASE_WATCHDOG.json').write_text(json.dumps({'deadline_epoch': DEADLINE, 'action': 'unassign own runtime'}) + '\n')

script = r'''
from pathlib import Path
import subprocess,sys,json,os,time,traceback
r=Path.home()/'logika-sixhour-broad-20261028'
try:
 assert time.time()<DEADLINE_VALUE
 subprocess.run(['/usr/local/bin/uv','venv','--python','3.11','--seed',str(r/'env311')],check=True,timeout=300)
 py=str(r/'env311/bin/python')
 subprocess.run([py,'-m','pip','install','--quiet','--index-url','https://download.pytorch.org/whl/cu126','torch==2.7.1'],check=True,timeout=1200)
 subprocess.run([py,'-m','pip','install','--quiet','structured-data-models==0.2.1','numpy==2.2.6','pandas==2.3.3','pyarrow==24.0.0','scipy==1.16.3','scikit-learn==1.7.2','lightgbm==4.6.0','catboost==1.2.8','xgboost==3.2.0','tabm==0.0.3','rtdl-num-embeddings==0.0.12','huggingface-hub==0.35.3','safetensors==0.6.2','typing_extensions==4.16.0'],check=True,timeout=900)
 inspect=r"""
from pathlib import Path
import hashlib,importlib.metadata as m,json,os,sys,torch
from huggingface_hub import hf_hub_download
r=Path.home()/'logika-sixhour-broad-20261028'
assert torch.cuda.is_available()
out={'status':'PASS','GPU':torch.cuda.get_device_name(0),'GPU_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'cores':os.cpu_count(),'CUDA':torch.version.cuda,'versions':{k:m.version(k)for k in ['numpy','pandas','pyarrow','scipy','scikit-learn','lightgbm','catboost','xgboost','torch','tabm','rtdl-num-embeddings','structured-data-models']},'fitted_weights_saved':False,'KV_caches_saved':False}
for size,ref in CHECKPOINT_VALUES.items():
 p=Path(hf_hub_download(repo_id=ref['repo_id'],filename=ref['filename'],revision=ref['revision'],token=False,cache_dir=str(r/'pretrained-cache')))
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 assert h.hexdigest()==ref['sha256']
 out['checkpoint_'+size]=str(p);out['checkpoint_'+size+'_sha256']=h.hexdigest()
(r/'ENVIRONMENT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
""".replace('CHECKPOINT_VALUES',repr(CHECKPOINT_RECORD))
 subprocess.run([py,'-c',inspect],check=True,timeout=600,env={**os.environ,'HF_HUB_OFFLINE':'0','HF_HUB_DISABLE_TELEMETRY':'1'})
except BaseException:
 (r/'SETUP_FAILURE.json').write_text(json.dumps({'error':traceback.format_exc(),'epoch':time.time()},indent=2)+'\n');raise
'''.replace('DEADLINE_VALUE', repr(DEADLINE)).replace('CHECKPOINT_RECORD', repr(CHECKPOINTS))
path = R / 'bootstrap.py'
if not path.exists():
    path.write_text(script)
    with (R / 'setup.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, str(path)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (R / 'BOOTSTRAP_DISPATCH.json').write_text(json.dumps({'pid': process.pid, 'started_epoch': time.time()}) + '\n')
print(json.dumps({'root': str(R), 'setup_dispatched': True, 'deadline_epoch': DEADLINE}), flush=True)
