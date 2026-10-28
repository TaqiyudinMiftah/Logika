"""PC2 control implementation; all PC3 access originates here."""
import argparse,json,shlex,subprocess,sys,time
from pathlib import Path
from supervisor import ROOT,read,write,rpc,remote,state,collect,report_all
from service import start
p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('host',default='all');p.add_argument('--job');p.add_argument('--output');a=p.parse_args();cfg=read(ROOT/'deployment.json');hosts=['pc1','pc2','pc3','node380'] if a.host=='all' else [a.host]
if a.action=='resume':
 assert time.time()<read(ROOT/'clock.json')['deadline'],'Original deadline expired; a new campaign requires a new explicit instruction'
 (ROOT/'operator_stop.json').unlink(missing_ok=True)
 if (ROOT/'finished.json').exists():(ROOT/'finished.json').rename(ROOT/'logs'/('previous-stop-'+str(int(time.time()))+'.json'))
 start('supervisor')
if a.action=='stop' and a.host=='all':write(ROOT/'operator_stop.json',{'time':time.time()})
if a.action=='launch':
 subprocess.run([sys.executable,str(ROOT/'launch.py')],check=True);raise SystemExit
for h in hosts:
 try:
  c=cfg['hosts'][h];base=c['root'];python=base+'/.venv/bin/python';runtime=base+'/campaign/runtime/campaign.py'
  if a.action=='status':
   s=state(h);counts={x:sum(j['status']==x for j in s.get('jobs',{}).values()) for x in set(j['status'] for j in s.get('jobs',{}).values())};print(json.dumps({'host':h,**{k:v for k,v in s.items() if k not in ['jobs','admission_failures']},'admission_wait_events':len(s.get('admission_failures',[])),'counts':counts}));continue
  if a.action in ['collect','report']:
   s=state(h);collect(h,s)
   if a.action=='report':report_all({x:state(x) for x in hosts})
   print(h,a.action,'done');continue
  if h=='pc1' and a.action=='stop':print(h,rpc({'action':'stop'}).decode());continue
  if h=='pc1':
   print(h,rpc({'action':'control','command':a.action,'job':a.job,'output':a.output}).decode());continue
  if a.action=='prepare':cmd=[python,base+'/host_prepare.py',h]
  elif a.action=='validate':cmd=[python,runtime,'validate',base+'/campaign']
  elif a.action=='stop':cmd=[python,runtime,'stop',base+'/campaign']
  elif a.action=='resume':
   if h in ['pc2','pc3']:cmd=[python,'-c',"import sys;sys.path.insert(0,"+repr(base)+");from service import start;print(start("+repr(h)+"))"]
   else:
    from node_join import client
    cnode,kernel=client();snippet='from pathlib import Path;import subprocess,json;r=Path('+repr(base)+');p=subprocess.Popen(['+repr(python)+','+repr(runtime)+',"resume",str(r/"campaign")],cwd=r,stdout=(r/"logs/resume.log").open("ab"),stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({"controller_pid":p.pid,"deadline_preserved":True}))';code,output=cnode.run_python(kernel,snippet,timeout=15);assert code==0,output;print(h,output);continue
  elif a.action=='verify':
   assert a.job;cmd=[python,runtime,'verify',base+'/campaign',a.job]
  elif a.action=='replay':
   assert a.job and a.output;cmd=[python,runtime,'replay',base+'/campaign',a.job,'--output',a.output]
  else:raise ValueError(a.action)
  print(h,remote(h,shlex.join(cmd),timeout=40))
 except Exception as e:print(json.dumps({'host':h,'error':str(e)[-400:]}))
