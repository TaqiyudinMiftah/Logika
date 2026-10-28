from pathlib import Path
import sys,tarfile,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent))
from connection_profiles import activated,load_profile
archive=ROOT/'stage/fm_source.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name in ['fm-source','evidence','toolkit','prepare_fm.py']:
  for p in ([ROOT/name] if (ROOT/name).is_file() else (ROOT/name).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
with activated(load_profile('node380')) as r:
 c=r.JupyterClient(r.DEFAULT_BASE_URL,r.load_token(Path(r.DEFAULT_SYSINFO)))
 c.push_file(archive,'faris/depth-20261017/fm_source.tar.gz','python3')
 text="""from pathlib import Path
import hashlib,tarfile,subprocess
r=Path('/home/taqiyudinmiftah/faris/depth-20261017')
assert hashlib.sha256((r/'fm_source.tar.gz').read_bytes()).hexdigest()==ARCHIVE_HASH
with tarfile.open(r/'fm_source.tar.gz') as t:
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())
 t.extractall(r)
(r/'env/lib/python3.11/site-packages/kumo-parent.pth').write_text('/home/taqiyudinmiftah/faris/kumo-large-node380-20261016/.venv/lib/python3.11/site-packages'+chr(10))
import os
os.environ['HF_HOME']='/home/taqiyudinmiftah/faris/kumo-large-node380-20261016/hf_cache'
os.environ['HF_HUB_OFFLINE']='1'
p=subprocess.Popen([str(r/'env/bin/python'),str(r/'fm-source/gpu_synthetic_test.py')],stdout=(r/'gpu-synthetic.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
""".replace('ARCHIVE_HASH',repr(hashlib.sha256(archive.read_bytes()).hexdigest()))
 code,out=c.run_python(c.session_kernel('term'),text,timeout=30);print(code,out)
