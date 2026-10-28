"""Recreate a collected transient unit without changing the campaign clock."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def start(host):
 clock=json.loads((ROOT/'clock.json').read_text());assert time.time()<clock['deadline'],'Original deadline expired'
 supervisor=host=='supervisor';name='explore12h-'+host;unit=name+'.service'
 if subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0:return 'already active'
 loaded=subprocess.run(['systemctl','--user','show',unit,'--property=LoadState','--value'],capture_output=True,text=True).stdout.strip()
 if loaded and loaded!='not-found':subprocess.run(['systemctl','--user','restart',unit],check=True);return 'restarted'
 h=json.loads((ROOT/'deployment.json').read_text())['hosts']['pc2' if supervisor else host];root=Path(h['root'])
 command=['systemd-run','--user','--slice=explore12h.slice','--unit='+name,'--property=Restart=on-failure','--property=RestartSec=15','--property=WorkingDirectory='+str(root),'--property=StandardOutput=append:'+str(root/'logs'/('supervisor.log' if supervisor else 'controller.log')),'--property=StandardError=append:'+str(root/'logs'/('supervisor.log' if supervisor else 'controller.log'))]
 if not supervisor:command+=['--property=CPUQuota='+str(h['cpus']*100)+'%','--property=MemoryMax='+str(h['ram_mb'])+'M','--property=MemorySwapMax=0']
 command += [str(root/'.venv/bin/python'),str(root/('supervisor.py' if supervisor else 'bootstrap.py'))]
 if not supervisor:command+=[host]
 subprocess.run(command,check=True);return 'recreated preserving original clock'
