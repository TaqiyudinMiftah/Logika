"""Explicit user-invoked operations only. No polling or automatic monitoring."""
import argparse,json,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent));sys.path.insert(0,str(ROOT.parent/'logika-meta-experiments'))
from metaexp.deployment import remote,machine_config
p=argparse.ArgumentParser();p.add_argument('action',choices=['status','stop','report','resume']);p.add_argument('host',choices=['pc1','pc1-amd','pc2','pc3','node380','all']);a=p.parse_args()
for h in (['pc1','pc1-amd','pc2','pc3','node380'] if a.host=='all' else [a.host]):
 if h=='node380':
  from connection_profiles import activated,load_profile
  with activated(load_profile(h)) as r:
   c=r.JupyterClient(r.DEFAULT_BASE_URL,r.load_token(Path(r.DEFAULT_SYSINFO)))
   root='/home/taqiyudinmiftah/faris/depth-20261017';cmd=[root+'/env/bin/python',root+'/campaign-fm2/runtime/campaign.py',a.action,root+'/campaign-fm2']
   if a.action=='resume':cmd+=['--detach']
   code,out=c.run_python(c.session_kernel('term'),'import subprocess; p=subprocess.run('+repr(cmd)+',capture_output=True,text=True);print(p.stdout);print(p.stderr)',timeout=40);print(h,out)
 else:
  target='pc1' if h=='pc1-amd' else h
  root=machine_config()['machines'][target]['root'].replace('logika-meta-experiments/meta-20261016-v1','depth-20261017')
  campaign='campaign-amd' if h=='pc1-amd' else ('campaign-parallel' if h in ['pc2','pc3'] else 'campaign')
  cmd=[root+'/.venv/bin/python',root+'/'+campaign+'/runtime/campaign.py',a.action,root+'/'+campaign]
  if a.action=='resume':cmd+=['--detach']
  print(h,remote(target,shlex.join(cmd),capture_output=True,text=True,timeout=50).stdout)
