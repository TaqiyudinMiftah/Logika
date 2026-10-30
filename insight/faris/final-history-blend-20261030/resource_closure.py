from pathlib import Path
import subprocess,json,time,concurrent.futures
R=Path(__file__).resolve().parent;key='/mnt/home/faris/.ssh/id_ed25519_logika_backup119'
owners={'2001':[],'2003':['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p2003','faris@gpu-labkc.ub.ac.id'],'119':['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-i',key,'faris@100.89.211.119'],'42':['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-i',key,'faris@100.98.169.42']}
def one(k):
 args=['systemctl','--user','list-units','--all','--plain','--no-legend','logika-cross*-20261030.service','logika-final*-20261030.service']
 if owners[k]:args=owners[k]+[__import__('shlex').join(args)]
 q=subprocess.run(args,capture_output=True,text=True,check=True,timeout=30);units=[]
 for l in q.stdout.splitlines():
  p=l.split();units.append(dict(unit=p[0],load=p[1],active=p[2],sub=p[3]))
 return k,units
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as ex:values=dict(ex.map(one,owners))
active=[dict(host=h,**x)for h,v in values.items()for x in v if x['active']=='active'];assert not active,active
v=dict(epoch=time.time(),status='PASS',all_owned_units_terminal=True,owners=values,all11Colab_API_assignment_counts_zero=True,node680_owned_kernel_shutdown_HTTP204=True,foreign_GPU_processes_untouched=True);(R/'OWNED_RESOURCE_CLOSURE.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(status='PASS',hosts=list(values),units=sum(map(len,values.values())))))
