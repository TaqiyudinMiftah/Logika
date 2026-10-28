"""Export small immutable evidence; retain all actual fitted models on node380."""
from pathlib import Path
import json,hashlib,tarfile,sys
r=Path(sys.argv[1]);staged=r.parent/'kumo-temperature-adaptation-20261021/full_data'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
plan=json.loads((r/'plan.json').read_text());weight=json.loads((r/'WEIGHT.json').read_text());out=r/'exports/heldout_fusion';out.mkdir(parents=True,exist_ok=True)
for f in range(5):
 d=r/f'folds/fold{f}';c=json.loads((d/'complete.json').read_text());assert c['status']=='complete'and c['full_replay_max_error']<2e-6
 m=json.loads((staged/f'fold{f}/manifest.json').read_text())
 for n,h in m['files'].items():assert sha(staged/f'fold{f}'/n)==h
 assert(d/'fitted.pkl').exists()
 assert(d/'fitted.pkl').stat().st_mtime_ns<=(d/'complete.json').stat().st_mtime_ns
 dest=out/f'fold{f}';dest.mkdir(exist_ok=True)
 for n in ['raw.npz','validation.npz','complete.json','boundary.json']:
  if n!='complete.json':assert sha(d/n)==c['files'][n]
  target=dest/n
  if target.exists():assert sha(target)==sha(d/n)
  else:target.write_bytes((d/n).read_bytes())
 receipt=dict(input_manifest=m,checkpoint_sha256=weight['sha256'],retained_model_path=str(d/'fitted.pkl'),
  retained_model_sha256_at_completion=c['files']['fitted.pkl'],model_size_bytes=(d/'fitted.pkl').stat().st_size,
  full_prediction_replay_error=c['full_replay_max_error'],source_study=str(r),source_hashes=plan['source_sha256'],model_not_transferred=True)
 (dest/'source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
a=r/'exports/heldout_fusion.tar.gz';assert not a.exists()
with tarfile.open(a,'w:gz')as t:
 for f in range(5):t.add(out/f'fold{f}',arcname=f'fold{f}')
assert a.stat().st_size<8*1024**2
(r/'exports/EXPORT.json').write_text(json.dumps(dict(status='complete',sha256=sha(a),bytes=a.stat().st_size,folds=5,model_weights_retained_remotely=True),indent=2)+'\n')
print('EXPORT_COMPLETE',a.stat().st_size,flush=True)
