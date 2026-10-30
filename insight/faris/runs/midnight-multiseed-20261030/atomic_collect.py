from pathlib import Path
import argparse,json,hashlib,shutil,os,time
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--owner',required=True);p.add_argument('--source',required=True);a=p.parse_args();src=Path(a.source);n=0
for phase in (src/'runs').glob('*/*'):
 if not phase.is_dir():continue
 rel=phase.relative_to(src);out=R/rel
 if (phase/'RESULT.json').exists():
  j=json.loads((phase/'RESULT.json').read_text())
  if not(phase/'START.json').exists()or not(phase/'prediction.npz').exists():continue
  assert sha(phase/'START.json')==j['start_sha256']and sha(phase/'prediction.npz')==j['prediction_sha256']
  if (out/'RESULT.json').exists():assert sha(out/'RESULT.json')==sha(phase/'RESULT.json');n+=1;continue
  temp=out.with_name(out.name+'.collecting-'+a.owner);temp.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(phase,temp,dirs_exist_ok=True)
  if out.exists():
   assert not(out/'RESULT.json').exists();archive=R/'private/collection-earlier'/a.owner/rel;archive.parent.mkdir(parents=True,exist_ok=True);out.rename(archive)
  temp.rename(out);n+=1
 elif any((phase/x).exists()for x in ['FAILURE.json','INTERRUPTION.json']):
  out.mkdir(parents=True,exist_ok=True)
  for x in ['START.json','FAILURE.json','INTERRUPTION.json']:
   if (phase/x).exists():
    if (out/x).exists():assert sha(out/x)==sha(phase/x)
    else:shutil.copy2(phase/x,out/x)
for p in src.glob('*.json'):
 target=R/('COLLECTED_'+a.owner+'_'+p.name);shutil.copy2(p,target)
(R/('COLLECTION_'+a.owner+'.json')).write_text(json.dumps(dict(epoch=time.time(),completed_fits=n,atomic_prediction_result_publication=True))+'\n');print(n)
