from pathlib import Path
import sys,tarfile,json,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent))
from connection_profiles import activated,load_profile
REL='faris/depth-20261017'
archive=ROOT/'stage/node_source.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name in ['source','evidence','toolkit','prepare.py']:
  for p in ([ROOT/name] if (ROOT/name).is_file() else (ROOT/name).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
with activated(load_profile('node380')) as r:
 c=r.JupyterClient(r.DEFAULT_BASE_URL,r.load_token(Path(r.DEFAULT_SYSINFO)))
 c.push_file(archive,REL+'/node_source.tar.gz','python3')
 code,out=c.run_python(c.session_kernel('term'),"from pathlib import Path\nimport tarfile,subprocess,hashlib\nr=Path('/home/taqiyudinmiftah/faris/depth-20261017')\nassert hashlib.sha256((r/'node_source.tar.gz').read_bytes()).hexdigest()=="+repr(hashlib.sha256(archive.read_bytes()).hexdigest())+"\nwith tarfile.open(r/'node_source.tar.gz') as t:\n assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())\n t.extractall(r)\np=subprocess.run([str(r/'env/bin/python'),str(r/'source/synthetic_test.py')],capture_output=True,text=True);print(p.returncode,p.stdout,p.stderr)",timeout=45)
 print(code,out)
