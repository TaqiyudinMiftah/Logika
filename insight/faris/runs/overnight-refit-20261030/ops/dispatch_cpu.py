from pathlib import Path
import json,subprocess,time
R=Path('/mnt/home/faris/logika-overnight-refit-20261030');KEY='/mnt/home/faris/.ssh/id_ed25519_logika_backup119'
OWNERS=[('2003',['ssh','-p','2003','faris@gpu-labkc.ub.ac.id'],'/mnt/home/faris','/mnt/home/faris/logika-submit-best-sub25k-20261029/env_gpu/bin/python',1,4),('119',['ssh','-i',KEY,'faris@100.89.211.119'],'/home/faris','/home/faris/logika-next-neural-cann-20261029/env311/bin/python',2,6),('42',['ssh','-i',KEY,'faris@100.98.169.42'],'/home/faris','/home/faris/logika-next-neural-cann-20261029/env311/bin/python',3,6)]
for name,ssh,home,py,owner,slots in OWNERS:
 dest=home+'/logika-overnight-refit-20261030';subprocess.run([*ssh,'mkdir -p '+dest+'/source '+dest+'/ops '+dest+'/packets'],check=True,timeout=30)
 for folder in ['source','packets']:
  subprocess.run(['rsync','-az','-e',' '.join(ssh[:-1]),str(R/folder)+'/',ssh[-1]+':'+dest+'/'+folder+'/'],check=True,timeout=900)
 for n in ['CLOCK.json','CONFIG.json','ops/local_queue.py']:
  subprocess.run([*ssh,'cat > '+dest+'/'+n],input=(R/n).read_bytes(),check=True,timeout=30)
 subprocess.run([*ssh,'systemd-run --user --unit=logika-0630-cann'+name+'-20261030 --property=MemoryMax=20G --property=CPUQuota='+str(slots*200)+'% --property=RuntimeMaxSec=4000 '+py+' -u '+dest+'/ops/local_queue.py --owner '+str(owner)+' --owners 4 --slots '+str(slots)],check=True,timeout=30)
print('All four CPU owners dispatched',flush=True)
