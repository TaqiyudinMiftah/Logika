"""Finite data-to-model dependency for the two predeclared feature conditions."""
from pathlib import Path
import json,time,subprocess,sys,traceback
R=Path(__file__).resolve().parent
def write(d,name):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 p=json.loads((R/'plan.json').read_text());deadline=p['original_deadline'];assert not(R/'MODEL_LAUNCH.json').exists()
 while time.time()<deadline-10800:
  if(R/'STOP').exists():return
  if(R/'BUILD_COMPLETE.json').exists():break
  status=subprocess.check_output(['systemctl','--user','show','logika-shopping-bands-build-20261021.service','--property=ActiveState','--value'],text=True).strip()
  if status not in ['active','activating']:raise RuntimeError('Feature build terminal without completion; inspect logs, no automatic retry.')
  time.sleep(10)
 else:raise RuntimeError('Original admission deadline reached.')
 audit=json.loads((R/'BUILD_COMPLETE.json').read_text());assert audit['default_labels_opened']is False and max(audit['reference_errors'].values())<=1e-6
 p['classifier_training_admitted']=True;write(p,'plan.json')
 with(R/'training.log').open('ab')as log:
  proc=subprocess.Popen([sys.executable,'-u',str(R/'source/run.py')],stdout=log,stderr=subprocess.STDOUT);write(dict(pid=proc.pid,original_deadline=deadline,feature_conditions=['bands','shuffled_bands']), 'MODEL_LAUNCH.json');code=proc.wait()
  if code:raise RuntimeError('Matched comparison failed; evidence retained, no automatic retry.')
 result=json.loads((R/'result.json').read_text());assert result['status']=='complete';write(dict(status='complete',original_deadline=deadline,goal_point_reached=result['goal_point_reached'],confirmation_opened=False,submission_generated=False),'STAGE_STATUS.json')
if __name__=='__main__':
 try:main()
 except BaseException:write(dict(status='failed',error=traceback.format_exc()),'STAGE_STATUS.json');raise
