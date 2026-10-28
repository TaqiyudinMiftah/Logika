"""Create a private, isolated execution backend inside the original campaign clock."""
from pathlib import Path
import json
import subprocess
import sys
import threading
import time

ROOT = Path.home() / 'logika-feature-subspace-moe-20261028'
ROOT.mkdir(parents=True, exist_ok=True)
DEADLINE = 1793230117.3252351
assert time.time() < DEADLINE - 7200

def release_at_original_deadline():
    while time.time() < DEADLINE:
        time.sleep(min(30, DEADLINE - time.time()))
    from google.colab import runtime
    runtime.unassign()

if not (ROOT / 'RELEASE_WATCHDOG.json').exists():
    threading.Thread(target=release_at_original_deadline, daemon=True).start()
    (ROOT / 'RELEASE_WATCHDOG.json').write_text(json.dumps({
        'deadline_epoch': DEADLINE, 'action': 'unassign this owned runtime',
        'original_clock_sha256': '693c430a2cdfe6cfc6c540e7e1e9480a0ae8944763b4cb1598da90b673456bac'
    }, indent=2) + '\n')

bootstrap = r'''
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
root=Path.home()/'logika-feature-subspace-moe-20261028'
try:
 assert time.time()<1793230117.3252351-7200
 subprocess.run([sys.executable,'-m','venv',str(root/'env')],check=True,timeout=120)
 python=str(root/'env/bin/python')
 subprocess.run([python,'-m','pip','install','--quiet','--index-url','https://download.pytorch.org/whl/cu126','torch==2.7.1'],check=True,timeout=1200)
 subprocess.run([python,'-m','pip','install','--quiet','structured-data-models==0.2.1','numpy==2.2.6','pandas==2.3.3','pyarrow==24.0.0','scipy==1.16.3','scikit-learn==1.7.2','lightgbm==4.6.0','catboost==1.2.8','huggingface-hub==0.35.3','safetensors==0.6.2','typing_extensions==4.16.0'],check=True,timeout=900)
 inspect=r"""
from pathlib import Path
import hashlib,importlib.metadata as m,json,sys,torch
from huggingface_hub import hf_hub_download
root=Path.home()/'logika-feature-subspace-moe-20261028'
p=Path(hf_hub_download(repo_id='nvidia/Kumo-Tabular',filename='large/classifier.pt',revision='v1.0.1',token=False,cache_dir=str(root/'pretrained-cache')))
h=hashlib.sha256()
with p.open('rb')as f:
 for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
assert h.hexdigest()=='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
record={'status':'PASS','checkpoint':str(p),'checkpoint_sha256':h.hexdigest(),'python':sys.version,'python_executable':sys.executable,'versions':{k:m.version(k)for k in ['structured-data-models','torch','numpy','pandas','pyarrow','scipy','scikit-learn','lightgbm','catboost','huggingface-hub']},'GPU':torch.cuda.get_device_name(0),'GPU_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'CUDA':torch.version.cuda,'fitted_weights_serialized':False,'KV_caches_serialized':False}
(root/'COLAB_ENVIRONMENT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
"""
 subprocess.run([python,'-c',inspect],check=True,timeout=600,env={**os.environ,'HF_HUB_OFFLINE':'0','HF_HUB_DISABLE_TELEMETRY':'1'})
except BaseException:
 (root/'COLAB_SETUP_FAILURE.json').write_text(json.dumps({'error':traceback.format_exc(),'epoch':time.time()},indent=2)+'\n')
 raise
'''
script = ROOT / 'bootstrap_execution.py'
if not script.exists():
    script.write_text(bootstrap)
    with (ROOT / 'bootstrap.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, str(script)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (ROOT / 'BOOTSTRAP_PID.json').write_text(json.dumps({'pid': process.pid, 'started_epoch': time.time()}) + '\n')
print(json.dumps({'root': str(ROOT), 'bootstrap': json.loads((ROOT / 'BOOTSTRAP_PID.json').read_text()), 'release_deadline_epoch': DEADLINE}), flush=True)
