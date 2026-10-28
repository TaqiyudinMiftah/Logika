"""One launch, one shared immutable clock, detached ordinary processes."""
import os,sys,time,json,subprocess,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'));from utils import read,write
from supervisor import rpc,remote
cfg=read(ROOT/'deployment.json')
assert not (ROOT/'clock.json').exists(),'Already launched; use explicit resume preserving clock'
for h in ['pc1','pc2','pc3']:
 proof=read(ROOT/'evidence'/('preflight-'+h+'.json'));assert proof['passed'],h
 proof=read(ROOT/'evidence'/('freeze-'+h+'.json'));assert proof['passed'],h
 from utils import sha
 assert proof['source_manifest_sha256']==sha(ROOT/'source_manifest.json') and proof['management_manifest_sha256']==sha(ROOT/'management_manifest.json'),h
clock={'start':time.time(),'deadline':time.time()+43200,'budget_seconds':43200,'id':ROOT.name};write(ROOT/'clock.json',clock)
rpc({'action':'clock','clock':clock})
path=cfg['hosts']['pc3']['root']+'/clock.json';code='from pathlib import Path;import json;p=Path('+repr(path)+');assert not p.exists();p.write_text('+repr(json.dumps(clock))+')';remote('pc3','python3 -c '+shlex.quote(code))
rpc({'action':'start'})
for host in ['pc3','pc2']:
 h=cfg['hosts'][host];cmd=['systemd-run','--user','--slice=explore12h.slice','--unit=explore12h-'+host,'--property=Restart=on-failure','--property=RestartSec=15','--property=CPUQuota='+str(h['cpus']*100)+'%','--property=MemoryMax='+str(h['ram_mb'])+'M','--property=MemorySwapMax=0','--property=WorkingDirectory='+h['root'],'--property=StandardOutput=append:'+h['root']+'/logs/controller.log','--property=StandardError=append:'+h['root']+'/logs/controller.log',h['root']+'/.venv/bin/python',h['root']+'/bootstrap.py',host];remote(host,shlex.join(cmd))
subprocess.run(['systemd-run','--user','--slice=explore12h.slice','--unit=explore12h-supervisor','--property=Restart=on-failure','--property=RestartSec=15','--property=WorkingDirectory='+str(ROOT),'--property=StandardOutput=append:'+str(ROOT/'logs/supervisor.log'),'--property=StandardError=append:'+str(ROOT/'logs/supervisor.log'),sys.executable,str(ROOT/'supervisor.py')],check=True)
write(ROOT/'launched.json',{'clock':clock,'pc_controllers_started':True,'node380':'late join same deadline'});print(json.dumps(clock))
