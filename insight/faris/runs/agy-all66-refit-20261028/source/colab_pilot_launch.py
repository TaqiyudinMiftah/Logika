"""Start the declared two-pattern A100 pilot in our isolated runtime."""
from pathlib import Path
import os,json,subprocess,sys,tarfile,hashlib,datetime
root=Path('/content/logika-all66');import shutil
for name in ['engine_runtime.py','engine_packets.py']:shutil.copy2(Path('/content')/name,root/name)
environment=json.loads((root/'ENVIRONMENT.json').read_text());bundle=Path('/content/pilot_packets.tar.gz')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='49112b76e811ed74f3c75a046cd4e8c3ed452150908c45c7b95516532ff553d1'
with tarfile.open(bundle)as tar:
 for m in tar.getmembers():assert not Path(m.name).is_absolute()and'..'not in Path(m.name).parts and(m.isdir()or m.isfile()),m.name
 tar.extractall(root,filter='data')
assert not(root/'pilot_outputs').exists(),'Preserve prior attempts; do not overwrite'
command=[sys.executable,str(root/'engine_packets.py'),'--packets',str(root/'pilot_packets'),'--out',str(root/'pilot_outputs'),'--checkpoint',environment['checkpoint'],'--mem-fraction','0.90']
log=(root/'pilot-run.log').open('w');env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
proc=subprocess.Popen(['timeout','1800',*command],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
v={'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':proc.pid,'purpose':'Predeclared cross-backend execution replay only','parameters':command,'source_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in ['engine_packets.py','engine_runtime.py']},'pretrained_checkpoint_sha256':environment['checkpoint_sha256'],'no_fitted_weights_or_KV_cache_serialized':True}
(root/'PILOT_START.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
