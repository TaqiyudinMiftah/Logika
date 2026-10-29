"""Pinned dependencies in the newly authorized node's actual user home."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,urllib.request,zipfile,time,traceback
R=Path('/home/faris/logika-next-gpu-extra-20261029');(R/'private').mkdir(parents=True,exist_ok=True)
try:
 meta=json.load(urllib.request.urlopen('https://pypi.org/pypi/uv/0.10.0/json',timeout=60))
 entry=next(v for v in meta['urls']if'manylinux_2_17_x86_64'in v['filename'])
 wheel=R/'private/uv.whl';urllib.request.urlretrieve(entry['url'],wheel);assert hashlib.sha256(wheel.read_bytes()).hexdigest()==entry['digests']['sha256']
 target=R/'private/uv-bootstrap';target.mkdir(exist_ok=True)
 with zipfile.ZipFile(wheel)as z:
  assert all((target/n).resolve().is_relative_to(target.resolve())for n in z.namelist());z.extractall(target)
 uv=next(p for p in target.rglob('uv')if p.is_file());uv.chmod(0o700)
 subprocess.run([str(uv),'venv','--python','3.11','--seed',str(R/'env311')],check=True,timeout=300)
 py=str(R/'env311/bin/python')
 subprocess.run([str(uv),'pip','install','--python',py,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.7.1'],check=True,timeout=1200)
 subprocess.run([str(uv),'pip','install','--python',py,'structured-data-models==0.2.1','numpy==2.2.6','pandas==2.3.3','pyarrow==24.0.0','scipy==1.16.3','scikit-learn==1.7.2','huggingface-hub==0.35.3','safetensors==0.6.2','typing_extensions==4.16.0'],check=True,timeout=900)
 code=r'''
from pathlib import Path
import json,hashlib,torch,os
from huggingface_hub import hf_hub_download
r=Path('/home/faris/logika-next-gpu-extra-20261029');refs=json.load(open(r/'CHECKPOINTS.json'));assert torch.cuda.is_available()
out=dict(status='PASS',GPU=torch.cuda.get_device_name(),GPU_memory_bytes=torch.cuda.get_device_properties(0).total_memory,actual_home=str(Path.home()),torch=torch.__version__)
for size in ['medium','large']:
 ref=refs[size];p=Path(hf_hub_download(repo_id=ref['repo_id'],filename=ref['filename'],revision=ref['revision'],cache_dir=str(r/'private/pretrained-cache'),token=False));assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
 target=r/'pretrained'/(size+'.pt');target.parent.mkdir(exist_ok=True)
 if not target.exists():target.symlink_to(p)
 assert target.resolve().is_relative_to(Path.home());out[size+'_sha256']=ref['sha256']
(r/'ENVIRONMENT_READY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''
 subprocess.run([py,'-c',code],check=True,timeout=600,env={**os.environ,'HF_HUB_OFFLINE':'0','HF_HUB_DISABLE_TELEMETRY':'1'})
except BaseException:
 (R/'private/BOOTSTRAP_FAILURE.log').write_text(traceback.format_exc());raise
