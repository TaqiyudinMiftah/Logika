"""Fixed production spline fits; no adaptive selection and no serialized weights."""
from pathlib import Path
import sys,subprocess,json,datetime,os
R=Path(__file__).resolve().parents[1]
for fold in range(5):
 frame=R/f'features/fold{fold}/expanded.parquet';assert frame.exists()
 out=R/f'assembly/splines/fold{fold}/complete.json'
 if out.exists():
  assert json.loads(out.read_text())['status']=='passed';continue
 print(json.dumps({'stage':'production_splines_start','fold':fold,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
 subprocess.run([sys.executable,str(R/'source/assembly_runner.py'),'splines','--fold',str(fold),'--features',str(frame)],check=True,timeout=1800)
print(json.dumps({'stage':'PRODUCTION_SPLINES_COMPLETE'}),flush=True)
