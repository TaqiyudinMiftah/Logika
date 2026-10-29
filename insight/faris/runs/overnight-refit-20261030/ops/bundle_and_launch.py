from pathlib import Path
import json,subprocess,tarfile
R=Path('/mnt/home/faris/logika-overnight-refit-20261030');HOST='faris@100.89.211.119';KEY='/mnt/home/faris/.ssh/id_ed25519_logika_backup119';DEST='/home/faris/logika-overnight-refit-20261030'
assert json.load(open(R/'PREPARATION_AUDIT.json'))['status']=='PASS'
(R/'CHECKPOINTS.json').write_bytes(Path('/mnt/home/faris/logika-twelvehour-next-20261029/CHECKPOINTS.json').read_bytes())
(R/'ENVIRONMENT.json').write_text(json.dumps({'checkpoint_large':'/root/logika-overnight-refit-20261030/checkpoints/large.pt'}))
(R/'private').mkdir(exist_ok=True)
with tarfile.open(R/'private/REFIT_BUNDLE.tgz','w:gz',compresslevel=1)as t:
 for n in ['source','packets','CLOCK.json','CONFIG.json','ENVIRONMENT.json','CHECKPOINTS.json','ops/bootstrap_colab_owned.py']:t.add(R/n,arcname=n)
ssh=['ssh','-i',KEY,HOST];subprocess.run([*ssh,'mkdir -p '+DEST+'/private/colab'],check=True)
subprocess.run(['rsync','-az','-e','ssh -i '+KEY,str(R/'private/REFIT_BUNDLE.tgz'),HOST+':'+DEST+'/private/colab/REFIT_BUNDLE.tgz'],check=True,timeout=1200)
code="""from pathlib import Path
import json,subprocess
r=Path('/home/faris/logika-overnight-refit-20261030')
for n in range(1,12):
 d=r/f'private/colab/account{n}'
 if(d/'ALLOCATED_READY.json').exists()and not(d/'RELEASED.json').exists():
  subprocess.run(['systemd-run','--user','--unit',f'logika-0630-colab-a{n}-20261030','--property=MemoryMax=4G','--property=RuntimeMaxSec=18000','/usr/bin/python3','-u',str(r/'ops/colab_controller.py'),'--account',str(n)],check=True)
"""
subprocess.run([*ssh,'python3 -'],input=code,text=True,check=True);print('Owned granted GPU controllers dispatched',flush=True)
