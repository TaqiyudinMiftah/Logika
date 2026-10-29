#!/usr/bin/python3
"""Match only this campaign's Colab evidence upload and publish complete dirs.

All other rsync operations retain the system command unchanged. This changes
filesystem publication only, never arrays, model parameters, source or RNG.
"""
from pathlib import Path
import os,re,subprocess,sys
args=sys.argv[1:];dest='faris@100.68.29.126:/mnt/home/faris/logika-twelvehour-20261029/runs/'
match=re.fullmatch(r'/home/faris/logika-twelvehour-20261029/private/colab/account([1-6])/mirror/runs/',args[-2])if len(args)>=4 and args[:2]==['-az','--ignore-existing']and args[-1]==dest else None
if match is None:os.execv('/usr/bin/rsync',['/usr/bin/rsync',*args])
N=int(match.group(1));LAB='faris@100.68.29.126';R='/mnt/home/faris/logika-twelvehour-20261029';stage=R+'/private/colab-atomic-incoming/account'+str(N)+'/runs'
subprocess.run(['ssh',LAB,'mkdir -p '+stage],check=True)
subprocess.run(['/usr/bin/rsync',*args[:-1],LAB+':'+stage+'/'],check=True)
code='''from pathlib import Path
import hashlib,json
r=Path(ROOT);s=r/'private/colab-atomic-incoming'/('account'+str(OWNER))/'runs';published=0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for d in s.glob('*/attempt1'):
 task=d.parent.name;assert task.startswith('colab_a'+str(OWNER)+'_')
 if not(d/'RESULT.json').exists()and not(d/'FAILURE.json').exists():continue
 if(d/'RESULT.json').exists():
  v=json.load(open(d/'RESULT.json'));assert v['status']=='complete'and sha(d/'START.json')==v['start_sha256']and sha(d/'prediction.npz')==v['prediction_sha256']
 target=r/'runs'/task/'attempt1';target.parent.mkdir(exist_ok=True,parents=True)
 if(target/'RESULT.json').exists():
  assert(d/'RESULT.json').exists()and sha(target/'RESULT.json')==sha(d/'RESULT.json')and sha(target/'prediction.npz')==json.load(open(d/'RESULT.json'))['prediction_sha256'];continue
 if target.exists():
  archive=r/'private/colab-atomic-incoming'/('account'+str(OWNER))/'pre_atomic_partial'/task/'attempt1';archive.parent.mkdir(exist_ok=True,parents=True);assert not archive.exists();target.rename(archive)
 d.rename(target);published+=1
print(json.dumps({'owner':OWNER,'atomically_published':published,'models_unchanged':True}))
'''.replace('ROOT',repr(R)).replace('OWNER',str(N))
subprocess.run(['ssh',LAB,'python3 -'],input=code,text=True,check=True)
