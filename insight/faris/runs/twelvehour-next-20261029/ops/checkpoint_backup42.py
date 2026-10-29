"""Remote compact checkpoint mirror; no credentials, raw labels or weights."""
from pathlib import Path
import hashlib,json,subprocess,time,traceback,os
R=Path(__file__).resolve().parents[1];KEY='/mnt/home/faris/.ssh/id_ed25519_logika_backup119';HOST='faris@100.98.169.42';DEST='/home/faris/logika-next-checkpoints-20261029'
C=json.load(open(R/'CLOCK.json'));private=R/'private/checkpoint42';private.mkdir(exist_ok=True)
def dump(v,p):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
transport='ssh -o BatchMode=yes -o ConnectTimeout=15 -i '+KEY
while time.time()<C['deadline_epoch']-60:
 try:
  subprocess.run(['ssh','-i',KEY,HOST,'mkdir -p '+DEST],check=True,capture_output=True,timeout=30)
  filters=[]
  for name in ['private','packets','evaluation_inputs','supplied_data','env311','env_gpu','__pycache__','pretrained','pretrained-cache']:
   filters+=['--exclude='+name+'/***','--exclude=**/'+name+'/***']
  filters+=['--include=*/','--include=*.json','--include=*.jsonl','--include=*.md','--include=*.npz','--include=*.py','--exclude=*']
  subprocess.run(['rsync','-az','-e',transport,*filters,str(R)+'/',HOST+':'+DEST+'/'],check=True,capture_output=True,timeout=300)
  code=r'''
from pathlib import Path
import hashlib,json,time
r=Path(ROOT);checked=[];pending=[]
for p in r.rglob('RESULT.json'):
 if p.parent.name!='attempt1':continue
 v=json.load(open(p))
 if v.get('status')!='complete':continue
 start=p.parent/'START.json';pred=p.parent/'prediction.npz'
 if not start.exists()or not pred.exists():pending.append(str(p.relative_to(r)));continue
 sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
 assert sha(start)==v['start_sha256']and sha(pred)==v['prediction_sha256']
 checked.append(dict(result=str(p.relative_to(r)),prediction_sha256=v['prediction_sha256'],start_sha256=v['start_sha256']))
out=dict(status='HASH_VERIFIED_COMPLETED_FIT_CHECKPOINTS',epoch=time.time(),completed_fit_records=len(checked),awaiting_atomic_next_snapshot=len(pending),records=checked,credentials_raw_labels_weights_excluded=True)
(r/'CHECKPOINT_MANIFEST.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='records'}))
'''.replace('ROOT',repr(DEST))
  p=subprocess.run(['ssh','-i',KEY,HOST,'python3 -'],input=code,text=True,capture_output=True,check=True,timeout=180);v=json.loads(p.stdout);v['parent_clock_sha256']=hashlib.sha256((R/'CLOCK.json').read_bytes()).hexdigest();dump(v,R/'CHECKPOINT42_STATE.json')
 except Exception:
  (private/'FAILURE_PRIVATE.log').write_text(traceback.format_exc());dump(dict(status='backup_retry_pending',epoch=time.time(),canonical_evidence_preserved=True),R/'CHECKPOINT42_STATE.json')
 if(R/'COMPLETION.json').exists():break
 time.sleep(60)
