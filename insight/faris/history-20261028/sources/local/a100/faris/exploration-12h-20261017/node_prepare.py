"""Node-only preparation before admission, sharing the immutable PC launch deadline."""
import os,sys,time,subprocess,json,hashlib,shutil,site
from pathlib import Path
ROOT=Path(__file__).resolve().parent
clock=json.loads((ROOT/'clock.json').read_text());assert time.time()<clock['start']+9*3600
try:
 envroot=ROOT/'.venv';subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(envroot)],check=True);python=str(envroot/'bin/python')
 target=subprocess.check_output([python,'-c','import site;print(site.getsitepackages()[0])'],text=True).strip();oldpaths=[p for p in sys.path if p.endswith('site-packages') and Path(p).is_dir()];Path(target,'kumo-existing.pth').write_text('\n'.join(oldpaths)+'\n')
 packages=['numpy==1.26.4','pandas==1.5.3','scipy==1.14.1','scikit-learn==1.6.1','pyarrow==22.0.0','lightgbm==4.6.0','catboost==1.2.8','xgboost==3.0.5','joblib==1.4.2','optuna==4.5.0','psutil==7.2.2','interpret-core==0.7.5']
 subprocess.run([python,'-m','pip','install',*packages],check=True,timeout=max(1,clock['start']+9*3600-time.time()))
 subprocess.run([python,'-m','pip','install','tabicl'],check=False,timeout=max(1,clock['start']+9*3600-time.time()))
 code="from huggingface_hub import hf_hub_download;import shutil;shutil.copyfile(hf_hub_download('jingang/TabICL','tabicl-classifier-v2-20260212.ckpt',token=False),"+repr(str(ROOT/'weights/tabicl-classifier-v2-20260212.ckpt'))+")"
 subprocess.run([python,'-c',code],check=False,timeout=min(600,max(1,clock['start']+9*3600-time.time())))
 assets={str(p.relative_to(ROOT/'weights')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'weights').glob('*.ckpt')};(ROOT/'evidence/node-weights.json').write_text(json.dumps(assets,indent=2))
 # Preserve the old campaign while restoring the reserved arrays in a new bank view.
 oldbank=Path('/home/taqiyudinmiftah/faris/depth-20261017/bank');newbank=ROOT/'bank';newbank.mkdir(exist_ok=True)
 for path in oldbank.iterdir():
  if path.is_file() and not (newbank/path.name).exists():
   try:os.link(path,newbank/path.name)
   except OSError:shutil.copy2(path,newbank/path.name)
 for path in (ROOT/'reserved').glob('*.npy'):shutil.copy2(path,newbank/path.name)
 manifest=json.loads((newbank/'manifest.json').read_text());assert manifest['bank_id']==json.loads((ROOT/'evidence/champion.json').read_text())['components']['bank_id']
 for name,checksum in manifest['files'].items():assert hashlib.sha256((newbank/name).read_bytes()).hexdigest()==checksum,('Node bank changed',name)
 deployment=json.loads((ROOT/'deployment.json').read_text());deployment['hosts']['node380']['bank']=str(newbank);(ROOT/'deployment.json').write_text(json.dumps(deployment))
 capacity_code="import sys,json;sys.path.insert(0,"+repr(str(ROOT/'toolkit'))+");from oneshot.common import probe;print(json.dumps(probe("+repr(str(ROOT))+")))"
 capacity=json.loads(subprocess.check_output([python,'-c',capacity_code],text=True));limits=deployment['hosts']['node380'];limits['cpus']=min(limits['cpus'],capacity['cpus']);limits['ram_mb']=min(limits['ram_mb'],max(0,int(capacity['ram_available_mb'])-2048));assert capacity['gpus'] and limits['ram_mb']>=12000,'Insufficient node capacity; no admission';(ROOT/'deployment.json').write_text(json.dumps(deployment));(ROOT/'evidence/node-capacity.json').write_text(json.dumps(capacity))
 # Fresh actual GPU and effective-quota probe; do not assume host resources are container limits.
 code="import torch,json;from pathlib import Path;import os;assert torch.cuda.is_available();x=torch.randn(64,64,device='cuda',requires_grad=True);(x@x).sum().backward();print(json.dumps({'gpu':torch.cuda.get_device_name(),'cpus':os.cpu_count(),'cgroup':Path('/proc/self/cgroup').read_text(),'deadline':"+repr(clock['deadline'])+"}))"
 probe=subprocess.check_output([python,'-c',code],text=True);(ROOT/'evidence/node-ready.json').write_text(probe)
 os.environ.update(HF_HOME='/home/taqiyudinmiftah/faris/kumo-large-node380-20261016/hf_cache',HF_HUB_OFFLINE='1',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
 os.execv(python,[python,str(ROOT/'bootstrap.py'),'node380'])
except BaseException as e:
 (ROOT/'node-preparation-failed.json').write_text(json.dumps({'error':type(e).__name__+': '+str(e),'time':time.time(),'deadline':clock['deadline']}));raise
