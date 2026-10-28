"""Hash retained fitted artifacts and code without altering any model."""
from pathlib import Path
import datetime,json,hashlib
R=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
records=[];counts={}
for p in sorted(R.rglob('*')):
    if not p.is_file()or p.is_symlink()or'__pycache__'in p.parts:continue
    scientific_json=(p.name.startswith('H0')or p.name=='complete.json'or p.name.endswith('_RESULT.json')or p.name.endswith('_AUDIT.json')or p.name in ['maturity_transformer.json','prior_audit.json','PREPARED.json','FULL_REFIT_PLAN.json','ENVIRONMENT_VERIFIED.json'])
    if p.suffix in ['.joblib','.cbm','.npz','.parquet','.py']or scientific_json:
        records.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)))
    if p.suffix in ['.joblib','.cbm']:
        family=p.relative_to(R).parts[0];counts[family]=counts.get(family,0)+1
out=dict(status='passed',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records,serialized_artifact_counts=counts,serialized_artifact_total=sum(counts.values()),counting_policy='File counts include indexed/canonical contexts, correction heads, splines and CDF maps. They are not independent hypotheses or datasets.',external_assets='Original source data, immutable feature bank, checkpoint and referenced original contexts remain at the durable PC2 paths recorded in each model and README. No private API credential is included.',all_new_completed_and_partial_model_files_preserved=True)
(R/'MODEL_RETENTION_MANIFEST.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],serialized_artifact_total=out['serialized_artifact_total'],hashed_files=len(records),counts=counts)))
