"""Laptop-independent collection, current-only audit and publication to submission host."""
from pathlib import Path
import hashlib,json,subprocess,time,traceback
R=Path('/mnt/home/faris/logika-overnight-refit-20261030');KEY='/mnt/home/faris/.ssh/id_ed25519_logika_backup119';PY='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
OWNERS=[(['ssh','-p','2003','faris@gpu-labkc.ub.ac.id'],'/mnt/home/faris'),(['ssh','-i',KEY,'faris@100.89.211.119'],'/home/faris'),(['ssh','-i',KEY,'faris@100.98.169.42'],'/home/faris')];C=json.load(open(R/'CLOCK.json'));HOST='faris@100.89.211.119';DEST='/home/faris/logika-overnight-refit-20261030'
def run(args,**kw):return subprocess.run(args,check=True,timeout=kw.pop('timeout',120),capture_output=True,**kw)
last=None
while time.time()<C['deadline_epoch']-180:
 try:
  for ssh,home in OWNERS:
   run(['rsync','-az','--ignore-existing','-e',' '.join(ssh[:-1]),ssh[-1]+':'+home+'/logika-overnight-refit-20261030/predictions/',str(R/'predictions')+'/'],timeout=120)
  files=list((R/'predictions').glob('*/s*/fold*/RESULT.json'));stamp=len(files)
  if stamp!=last:
   q=run([PY,'-u',str(R/'source/fusion.py')],timeout=1200);(R/'FUSION_LAST_STDOUT.log').write_bytes(q.stdout);last=stamp
  if(R/'READY_CANDIDATE.json').exists():
   ready=json.load(open(R/'READY_CANDIDATE.json'));out=Path(ready['path']);audit=Path(ready['audit_path']);assert hashlib.sha256(out.read_bytes()).hexdigest()==ready['sha256']
   for p in [out,audit]:run(['scp','-q','-i',KEY,str(p),HOST+':'+DEST+'/'+p.name])
   ready['path']=DEST+'/'+out.name;ready['audit_path']=DEST+'/'+audit.name;payload=json.dumps(ready,indent=2)+'\n'
   run(['ssh','-i',KEY,HOST,'python3 -'],input=("from pathlib import Path;r=Path("+repr(DEST)+");p=r/'READY_CANDIDATE.tmp';p.write_text("+repr(payload)+");p.replace(r/'READY_CANDIDATE.json')").encode())
  (R/'COLLECTION_STATE.json').write_text(json.dumps(dict(epoch=time.time(),complete=stamp,current_candidate=(R/'READY_CANDIDATE.json').exists()))+'\n')
 except Exception:(R/'COLLECTION_REPAIR_PRIVATE.log').write_text(traceback.format_exc())
 time.sleep(30)
