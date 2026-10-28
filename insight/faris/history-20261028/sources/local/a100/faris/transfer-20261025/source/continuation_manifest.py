"""Create a new complete archive inventory without overwriting the previous seal."""
from pathlib import Path
import datetime
import hashlib
import json

R=Path(__file__).resolve().parents[1]
out=R/'CONTINUATION_RETENTION_MANIFEST.json'
assert not out.exists(), 'Preserve the existing manifest; use a new named snapshot.'
excluded={out.name,'CONTINUATION_LOCAL_DELIVERY_AUDIT.json'}
records=[];counts={}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
for p in sorted(R.rglob('*')):
    if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.name in excluded:continue
    records.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)))
    if p.suffix in ('.joblib','.cbm'):
        family=p.relative_to(R).parts[0];counts[family]=counts.get(family,0)+1
manifest=dict(status='passed',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              files=records,serialized_artifact_counts=counts,serialized_artifact_total=sum(counts.values()),
              scope='Every regular non-cache file in the campaign archive, including plans, parameters, logs, and prior seals.',
              exclusions=sorted(excluded)+['__pycache__ directories','symlinks'],
              external_assets='Original source data, feature bank, frozen checkpoint and referenced historical contexts remain at the durable PC2 paths in individual bundles. This is not a standalone distribution.',
              counting_policy='Serialized file count, including indexed models and fitted maps; not independent experiments.')
out.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(hashed_files=len(records),serialized_artifact_total=sum(counts.values()),
                      indexed_bytes=sum(x['bytes']for x in records),manifest_sha256=sha(out))))
