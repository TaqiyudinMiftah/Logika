from pathlib import Path
import json,time,subprocess,sys,traceback
R=Path(__file__).resolve().parent
def write(d):
 p=R/'FINAL_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 deadline=json.loads((R/'plan.json').read_text())['original_deadline']
 while time.time()<deadline:
  if(R/'STOP').exists():return
  p=R/'STAGE_STATUS.json'
  if p.exists():
   d=json.loads(p.read_text())
   if d['status']=='failed':raise RuntimeError('Feature/trainingchainfailed; inspectevidence,noautomaticretry.')
   if d['status']=='complete':break
  time.sleep(10)
 else:raise RuntimeError('Originaldeadlineexpired.')
 with(R/'verification.log').open('ab')as log:subprocess.run([sys.executable,str(R/'verify_models.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 audit=json.loads((R/'INDEPENDENT_AUDIT.json').read_text());assert audit['status']=='passed'and audit['models_replayed']==10
 result=json.loads((R/'result.json').read_text());write(dict(status='passed',models_replayed=10,metrics=audit['metrics'],goal_point_reached=result['goal_point_reached'],original_deadline=deadline,confirmation_opened=False,submission_generated=False))
if __name__=='__main__':
 try:main()
 except BaseException:write(dict(status='failed',error=traceback.format_exc()));raise
