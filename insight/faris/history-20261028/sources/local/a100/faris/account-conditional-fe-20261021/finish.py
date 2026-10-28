"""Finite verification dependency, never proposes or starts another experiment."""
from pathlib import Path
import json,sys,time,subprocess,traceback
R=Path(__file__).resolve().parent
def write(d):
 p=R/'FINAL_STATUS.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def main():
 deadline=json.loads((R/'plan.json').read_text())['original_deadline']
 while time.time()<deadline:
  if(R/'STOP').exists():return
  p=R/'STAGE_STATUS.json'
  if p.exists():
   d=json.loads(p.read_text())
   if d['status']=='failed':raise RuntimeError('Preparation/training chain failed; preserve incident and inspect logs, no automatic retry.')
   if d['status']=='complete':break
  time.sleep(10)
 else:raise RuntimeError('Original deadline expired before terminal report.')
 with(R/'model-verification.log').open('ab')as log:subprocess.run([sys.executable,str(R/'verify_models.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 evidence=json.loads((R/'INDEPENDENT_AUDIT.json').read_text());assert evidence['status']=='passed'and evidence['models_replayed']==10
 report=json.loads((R/'result.json').read_text());write(dict(status='passed',models_replayed=10,feature_gate_passed=True,metrics=evidence['metrics'],goal_point_reached=report['goal_point_reached'],original_deadline=deadline,confirmation_opened=False,submission_generated=False))
if __name__=='__main__':
 try:main()
 except BaseException:write(dict(status='failed',error=traceback.format_exc()));raise
