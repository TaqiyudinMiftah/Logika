from pathlib import Path
import json,tarfile,hashlib
r=Path('/content/logika-all66');complete=[];failures=[]
for p in (r/'predictions').rglob('COMPLETE.json'):
 v=json.loads(p.read_text());data=p.with_name('prediction.npz');assert hashlib.sha256(data.read_bytes()).hexdigest()==v['prediction_sha256'];complete.append(str(p.parent.relative_to(r/'predictions')))
for p in (r/'predictions').rglob('FAILURE.json'):failures.append(str(p))
for p in r.glob('production-*.log'):print(p.name,'\n','\n'.join(p.read_text(errors='replace').splitlines()[-3:]))
v={'complete':sorted(complete),'complete_count':len(complete),'failures':failures,'all_completed_prediction_hashes_verified':True};print('STATUS_JSON '+json.dumps(v),flush=True)
if len(complete)in[5,35,43]and not failures:
 out=r/f'results_{len(complete)}.tar.gz'
 with tarfile.open(out,'w:gz')as tar:
  for name in ['predictions','ENVIRONMENT.json']:tar.add(r/name,arcname=name)
  for p in r.glob('PRODUCTION_START_*.json'):tar.add(p,arcname=p.name)
  for p in r.glob('production-*.log'):tar.add(p,arcname=p.name)
 print('RESULT_BUNDLE '+str(out),flush=True)
