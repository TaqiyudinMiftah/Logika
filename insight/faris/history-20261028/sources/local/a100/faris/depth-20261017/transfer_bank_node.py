from pathlib import Path
import sys,tarfile,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent))
from connection_profiles import activated,load_profile
archive=ROOT/'stage/bank.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for p in (ROOT/'stage/bank').iterdir():
  if p.is_file():t.add(p,arcname='bank/'+p.name,recursive=False)
print('Archive bytes',archive.stat().st_size,flush=True)
with activated(load_profile('node380')) as r:
 c=r.JupyterClient(r.DEFAULT_BASE_URL,r.load_token(Path(r.DEFAULT_SYSINFO)))
 c.push_file(archive,'faris/depth-20261017/bank.tar.gz','python3')
 code,out=c.run_python(c.session_kernel('term'),"from pathlib import Path\nimport tarfile,hashlib\nr=Path('/home/taqiyudinmiftah/faris/depth-20261017')\nassert hashlib.sha256((r/'bank.tar.gz').read_bytes()).hexdigest()=="+repr(hashlib.sha256(archive.read_bytes()).hexdigest())+"\nwith tarfile.open(r/'bank.tar.gz') as t:\n assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())\n t.extractall(r)\nprint('BANK_TRANSFER_VERIFIED')",timeout=60)
 print(code,out)
