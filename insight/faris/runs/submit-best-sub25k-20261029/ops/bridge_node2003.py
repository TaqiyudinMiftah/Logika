"""Collect owned raw members from2003; assemble only against canonical references."""
from pathlib import Path
import json,subprocess,time,sys,hashlib,traceback
R=Path(__file__).resolve().parents[1];C=json.load(open(R/'DEPLOYMENT_CLOCK.json'));CPU='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
try:
 while time.time()<C['deadline_epoch']-10:
  (R/'predictions').mkdir(exist_ok=True)
  p=subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -p2003','faris@gpu-labkc.ub.ac.id:'+str(R)+'/predictions/',str(R/'predictions')+'/'],capture_output=True,text=True,timeout=120)
  assert p.returncode in[0,23],p.stderr
  for n in ['LIVE_STATE.json','DEPLOYMENT_RUNTIME_GATE.json','NODE2003_ENVIRONMENT.json','QUEUE_FAILURE.json']:
   subprocess.run(['rsync','-az','-e','ssh -p2003','faris@gpu-labkc.ub.ac.id:'+str(R/n),str(R)+'/'],capture_output=True,timeout=30)
  if(R/'QUEUE_FAILURE.json').exists():raise RuntimeError('Preserved remote worker failure; no automatic retry or upload')
  complete=list((R/'predictions').glob('s*/fold*/RESULT.json'))
  if len(complete)==10:
   for p in complete:
    v=json.load(open(p));assert hashlib.sha256((p.parent/'prediction.npz').read_bytes()).hexdigest()==v['prediction_sha256'];assert hashlib.sha256((p.parent/'START.json').read_bytes()).hexdigest()==v['start_sha256']
   subprocess.run([CPU,str(R/'source/assemble.py')],check=True,timeout=180);print('Ten raw members verified; canonical assembly complete',flush=True);break
  time.sleep(10)
 else:raise TimeoutError('Original fixed deployment deadline reached')
except Exception:
 (R/'private/BRIDGE_FAILURE_PRIVATE.log').write_text(traceback.format_exc());raise
