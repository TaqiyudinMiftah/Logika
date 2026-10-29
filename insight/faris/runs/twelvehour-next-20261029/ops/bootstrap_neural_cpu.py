"""New CPU-only Torch environment; preserve earlier campaign environments."""
from pathlib import Path
import subprocess,sys,json,shutil,time,os
R=Path('/home/faris/logika-next-neural-cann-20261029');R.mkdir(exist_ok=True)
assert shutil.disk_usage(R).free>1500*1024**2,'Leave existing data and512MiBreserve intact'
uv=next((p for p in [shutil.which('uv'),'/home/faris/.local/bin/uv','/home/faris/logika-next-cpu-exactks-20261029/private/uv'] if p and Path(p).is_file()),None)
assert uv,'No existing uv binary found; do not modify previous environments'
subprocess.run([uv,'venv','--python','3.11','--seed',str(R/'env311')],check=True,timeout=300)
py=str(R/'env311/bin/python')
subprocess.run([uv,'pip','install','--python',py,'--index-url','https://download.pytorch.org/whl/cpu','torch==2.7.1'],check=True,timeout=1200)
subprocess.run([uv,'pip','install','--python',py,'numpy==2.2.6','pandas==2.3.3','pyarrow==24.0.0','scipy==1.16.3','scikit-learn==1.7.2','threadpoolctl==3.6.0'],check=True,timeout=600)
code="import torch,json;from pathlib import Path;v={'status':'PASS','torch':torch.__version__,'CPU_only_requested':True};Path("+repr(str(R/'ENVIRONMENT_READY.json'))+").write_text(json.dumps(v)+'\\n');print(json.dumps(v))"
subprocess.run([py,'-c',code],check=True,timeout=90,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
