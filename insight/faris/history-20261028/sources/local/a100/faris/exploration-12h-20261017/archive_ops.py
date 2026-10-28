"""Delete only redundant local model copies after archive-side verification."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def prune(request):
 folder=(ROOT/request['directory']).resolve();assert folder.is_relative_to(ROOT/'campaign/runs')
 assert (folder/'outcome.json').exists();out=json.loads((folder/'outcome.json').read_text());assert out['status']=='complete'
 allowed=set(out['models']);receipt={'archive':request['archive'],'verified_files':request['files'],'directory':request['directory'],'bytes_removed':0,'reason':'Redundant local copy; durable PC2 archive retains identical bytes'}
 # Verify every requested source file before writing a durable receipt or removing any.
 for name,h in request['files'].items():
  assert out['artifacts'].get(name)==h
  assert name in allowed and name.endswith(('.joblib','.pt','.pkl'))
  p=(folder/name).resolve();assert p.is_relative_to(folder) and not (folder/name).is_symlink()
  if p.exists():assert sha(p)==h
 tmp=folder/'archive_receipt.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(folder/'archive_receipt.json')
 for name in request['files']:
  p=folder/name
  if p.exists():receipt['bytes_removed']+=p.stat().st_size;p.unlink()
 (folder/'archive_receipt.json').write_text(json.dumps(receipt,indent=2))
 with (ROOT/'logs/archive-pruning.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
 return receipt

if __name__=='__main__':print(json.dumps(prune(json.load(sys.stdin))))
