"""Prepare, seal and launch the one authorized node380 FM campaign. No polling."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent))
from connection_profiles import activated,load_profile
with activated(load_profile('node380')) as r:
 c=r.JupyterClient(r.DEFAULT_BASE_URL,r.load_token(Path(r.DEFAULT_SYSINFO)))
 text="""from pathlib import Path
import os,json,subprocess,time
r=Path('/home/taqiyudinmiftah/faris/depth-20261017')
assert (r/'bank/numeric.npy').exists(),'Feature transfer must finish first'
assert not (r/'campaign-fm/state.json').exists(),'Existing campaign: inspect; never reset clock'
os.environ.update(HF_HOME='/home/taqiyudinmiftah/faris/kumo-large-node380-20261016/hf_cache',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
script='set -e\\n'+str(r/'env/bin/python')+' '+str(r/'prepare_fm.py')+'\\nexec '+str(r/'env/bin/python')+' '+str(r/'campaign-fm/runtime/campaign.py')+' run '+str(r/'campaign-fm')
p=subprocess.Popen(['/bin/bash','-c',script],cwd=r,stdin=subprocess.DEVNULL,stdout=(r/'fm-controller.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':p.pid,'proc_start_ticks':Path('/proc/%s/stat'%p.pid).read_text().split()[21],'root':str(r),'started':time.time(),'absolute_deadline':1792191600}
(r/'fm-bootstrap.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
"""
 code,out=c.run_python(c.session_kernel('term'),text,timeout=20)
 (ROOT/'evidence/node-bootstrap.json').write_text(out)
 print(code,out)
