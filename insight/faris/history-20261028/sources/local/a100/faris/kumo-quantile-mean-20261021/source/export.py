from pathlib import Path
import tarfile,json,hashlib
import baseline as b
R=Path(__file__).resolve().parents[1]
assert json.loads((R/'STATUS.json').read_text())['status']=='training_complete_ready_for_PC2_scoring'
out=R/'exports';out.mkdir(exist_ok=True);archive=out/'evidence.tar.gz';assert not archive.exists()
receipts=[]
with tarfile.open(archive,'w:gz')as t:
 for f in range(5):
  d=R/f'folds/fold{f}';c=json.loads((d/'complete.json').read_text())
  for n,h in c['files'].items():assert b.sha(d/n)==h
  cache=json.loads((d/'cache.json').read_text());assert cache['sha256']==c['retained_cache_sha256']and(d/'fitted_context.pt').stat().st_size==cache['bytes']
  receipts.append(dict(fold=f,cache=cache,model_path=str(d/'fitted_context.pt'),quantiles_path=str(d/'raw_quantiles.npy'),quantiles_sha256=c['files']['raw_quantiles.npy'],large_output_hash_verification='Fittedcache bytes independentlyhashed and replayed duringfit; quantiles rereadhashed here. Models and quantilearrays retainedsource.'))
  for n in ['raw.npz','validation.npz','cache.json','boundary.json','correction.joblib','complete.json']:t.add(d/n,arcname=str((d/n).relative_to(R)))
 for n in ['WEIGHT.json','PREFLIGHT.json','plan.json','POINT_RESULT.json','OOF.npz','result.json']:
  if(R/n).exists():t.add(R/n,arcname=n)
 receipt=out/'LARGE_OUTPUTS.json';receipt.write_text(json.dumps(receipts,indent=2));t.add(receipt,arcname='LARGE_OUTPUTS.json')
assert archive.stat().st_size<8*1024**2
(out/'EXPORT.json').write_text(json.dumps(dict(status='complete',sha256=b.sha(archive),bytes=archive.stat().st_size),indent=2))
