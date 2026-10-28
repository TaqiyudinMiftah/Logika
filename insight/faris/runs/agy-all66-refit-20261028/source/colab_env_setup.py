"""Prepare only our account2 A100 runtime; exact pinned Engine dependencies.

Execute through `colab --account 2 exec -s OWN_SESSION --file THIS --timeout 900`.
Pretrained checkpoint caching is permitted; fitted weights/KV caches are not saved.
"""
from pathlib import Path
import subprocess,sys,json,hashlib,os
root=Path('/content/logika-all66');root.mkdir(parents=True,exist_ok=True)
# Fresh subprocesses avoid stale imports in the notebook kernel after pip changes.
subprocess.run([sys.executable,'-m','pip','install','--quiet','--index-url','https://download.pytorch.org/whl/cu126','torch==2.7.1'],check=True)
subprocess.run([sys.executable,'-m','pip','install','--quiet','structured-data-models==0.2.1','numpy==2.2.6','pandas==2.3.3','pyarrow==22.0.0','huggingface-hub==0.35.3'],check=True)
code=r'''
from pathlib import Path
import hashlib,json,sys,importlib.metadata as m
from huggingface_hub import hf_hub_download
import torch
root=Path('/content/logika-all66')
p=Path(hf_hub_download(repo_id='nvidia/Kumo-Tabular',filename='large/classifier.pt',revision='v1.0.1',token=False))
h=hashlib.sha256()
with p.open('rb')as f:
 for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
assert h.hexdigest()=='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
assert torch.cuda.is_available()
assert 'A100'in torch.cuda.get_device_name(0)
v={'checkpoint':str(p),'checkpoint_sha256':h.hexdigest(),'python':sys.version,'versions':{k:m.version(k)for k in ['structured-data-models','torch','numpy','pandas','pyarrow','huggingface-hub']},'GPU':torch.cuda.get_device_name(0),'GPU_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'CUDA':torch.version.cuda,'fitted_weights_serialized':False,'KV_caches_serialized':False}
(root/'ENVIRONMENT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
'''
subprocess.run([sys.executable,'-c',code],check=True,env={**os.environ,'HF_HUB_OFFLINE':'0','HF_HUB_DISABLE_TELEMETRY':'1'})
