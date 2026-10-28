from pathlib import Path
import sys,tarfile,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent))
from connection_profiles import activated,load_profile
archive=ROOT/'stage/fm2_source.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name in ['fm2-source','prepare_fm2.py']:
  for p in ([ROOT/name] if (ROOT/name).is_file() else (ROOT/name).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
with activated(load_profile('node380')) as r:
 c=r.JupyterClient(r.DEFAULT_BASE_URL,r.load_token(Path(r.DEFAULT_SYSINFO)))
 c.upload_small(archive,'faris/depth-20261017/fm2_source.tar.gz')
 text="""from pathlib import Path
import os,json,subprocess,time,tarfile,hashlib
r=Path('/home/taqiyudinmiftah/faris/depth-20261017')
assert hashlib.sha256((r/'fm2_source.tar.gz').read_bytes()).hexdigest()==ARCHIVE_HASH
with tarfile.open(r/'fm2_source.tar.gz') as t:
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(r)
assert not (r/'campaign-fm2').exists()
os.environ.update(HF_HOME='/home/taqiyudinmiftah/faris/kumo-large-node380-20261016/hf_cache',HF_HUB_OFFLINE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
script='set -e\\n'+str(r/'env/bin/python')+' '+str(r/'prepare_fm2.py')+'\\nexec '+str(r/'env/bin/python')+' '+str(r/'campaign-fm2/runtime/campaign.py')+' run '+str(r/'campaign-fm2')
p=subprocess.Popen(['/bin/bash','-c',script],cwd=r,stdin=subprocess.DEVNULL,stdout=(r/'fm2-controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':p.pid,'proc_start_ticks':Path('/proc/%s/stat'%p.pid).read_text().split()[21],'root':str(r),'started':time.time(),'absolute_deadline':1792191600}
(r/'fm2-bootstrap.json').write_text(json.dumps(record));print(json.dumps(record))
""".replace('ARCHIVE_HASH',repr(hashlib.sha256(archive.read_bytes()).hexdigest()))
 code,out=c.run_python(c.session_kernel('term'),text,timeout=20);print(code,out)
