"""Owned VM dependency setup only; source/inputs arrive separately before fitting."""
from pathlib import Path
import json,os,subprocess,sys,shutil,time,traceback
R=Path('/root/logika-twelvehour-next-20261029');R.mkdir(exist_ok=True)
deadline=json.loads((R/'CLOCK.json').read_text())['deadline_epoch'];assert time.time()<deadline
try:
    if shutil.which('uv')is None:subprocess.run([sys.executable,'-m','pip','install','--quiet','uv'],check=True,timeout=240)
    uv=shutil.which('uv');subprocess.run([uv,'venv','--python','3.11','--seed',str(R/'env311')],check=True,timeout=300)
    py=str(R/'env311/bin/python')
    subprocess.run([uv,'pip','install','--python',py,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.7.1'],check=True,timeout=1200)
    subprocess.run([uv,'pip','install','--python',py,'structured-data-models==0.2.1','numpy==2.2.6','pandas==2.3.3','pyarrow==24.0.0','scipy==1.16.3','scikit-learn==1.7.2','huggingface-hub==0.35.3','safetensors==0.6.2','typing_extensions==4.16.0'],check=True,timeout=900)
    code=r'''
from pathlib import Path
import json,hashlib,os,torch,importlib.metadata
from huggingface_hub import hf_hub_download
r=Path('/root/logika-twelvehour-next-20261029');assert torch.cuda.is_available()
env=json.load(open(r/'ENVIRONMENT.json'));refs=json.load(open(r/'CHECKPOINTS.json'))
out={'status':'PASS','GPU':torch.cuda.get_device_name(0),'GPU_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'cores':os.cpu_count(),'versions':{k:importlib.metadata.version(k)for k in ['torch','numpy','pandas','pyarrow','scipy','scikit-learn','structured-data-models']}}
for size in ['medium','large']:
 ref=refs[size];p=Path(hf_hub_download(repo_id=ref['repo_id'],filename=ref['filename'],revision=ref['revision'],token=False,cache_dir=str(r/'pretrained-cache')))
 assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
 # Preserve the sealed logical ENVIRONMENT path; physical checkpoint remains inside /root.
 logical=Path(env['checkpoint_'+size]);logical.parent.mkdir(parents=True,exist_ok=True)
 if not logical.exists():logical.symlink_to(p)
 assert logical.resolve().is_relative_to(Path.home())
 out['checkpoint_'+size+'_sha256']=ref['sha256']
(r/'ENVIRONMENT_LIVE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
'''
    subprocess.run([py,'-c',code],check=True,timeout=600,env={**os.environ,'HF_HUB_OFFLINE':'0','HF_HUB_DISABLE_TELEMETRY':'1'})
except Exception:
    (R/'SETUP_FAILURE.json').write_text(json.dumps(dict(UTC=time.time(),traceback=traceback.format_exc(),no_training_performed=True),indent=2)+'\n');raise
