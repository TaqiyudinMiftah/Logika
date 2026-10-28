"""Finite dependency chain for the one already specified feature hypothesis."""
from pathlib import Path
import sys,json,time,subprocess,traceback,hashlib
R=Path(__file__).resolve().parent
def write(d,p):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def main():
 plan=json.loads((R/'plan.json').read_text());deadline=plan['original_deadline'];assert not(R/'MODEL_LAUNCH.json').exists()
 while time.time()<deadline-10800:
  if(R/'STOP').exists():return
  if(R/'BUILD_COMPLETE.json').exists():break
  status=subprocess.check_output(['systemctl','--user','show','logika-account-conditional-build-20261021.service','--property=ActiveState','--value'],text=True).strip()
  if status not in ['active','activating']:raise RuntimeError('Feature preparation is terminal without completion; inspect build.log, do not restart blindly.')
  time.sleep(10)
 else:raise RuntimeError('Original admission deadline reached before feature readiness.')
 assert json.loads((R/'BUILD_COMPLETE.json').read_text())['default_labels_opened']is False
 with(R/'feature-verification.log').open('ab')as log:subprocess.run([sys.executable,str(R/'verify_features.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 assert json.loads((R/'INDEPENDENT_FEATURE_AUDIT.json').read_text())['status']=='passed'
 # Admission is an execution gate, not another feature/model recipe or clock.
 assert time.time()<deadline-10800
 plan['classifier_training_admitted']=True;plan['feature_verification_sha256']=hashlib.sha256((R/'INDEPENDENT_FEATURE_AUDIT.json').read_bytes()).hexdigest();write(plan,R/'plan.json')
 with(R/'training.log').open('ab')as log:
  p=subprocess.Popen([sys.executable,'-u',str(R/'source/run.py')],stdout=log,stderr=subprocess.STDOUT)
  write(dict(pid=p.pid,original_deadline=deadline,one_predeclared_feature_recipe=True),R/'MODEL_LAUNCH.json');code=p.wait()
  if code:raise RuntimeError('Fixed fivefold comparison failed; no automatic retry.')
 result=json.loads((R/'result.json').read_text());assert result['status']=='complete';write(dict(status='complete',original_deadline=deadline,goal_point_reached=result['goal_point_reached'],confirmation_opened=False,submission_generated=False),R/'STAGE_STATUS.json')
if __name__=='__main__':
 try:main()
 except BaseException:write(dict(status='failed',error=traceback.format_exc()),R/'STAGE_STATUS.json');raise
