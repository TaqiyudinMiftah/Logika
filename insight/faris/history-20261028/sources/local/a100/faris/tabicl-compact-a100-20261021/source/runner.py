"""Full-cache verification retry; original deadline and full folds preserved."""
import os,sys,subprocess,time,json,signal
from pathlib import Path
R=Path(__file__).resolve().parents[1];p=json.loads((R/'plan.json').read_text());D=p['original_deadline'];done=[]
def write(x):
 t=R/'STATUS.tmp';t.write_text(json.dumps(x,indent=2));t.replace(R/'STATUS.json')
assert p['training_admitted']and json.loads((R/'PREFLIGHT.json').read_text())['status']=='passed'
for f in range(5):
 if time.time()>=D-10800 or(R/'STOP').exists():write(dict(status='partial',completed=done,reason='Original admission deadline or stop'));sys.exit(1)
 assert not(R/f'folds/fold{f}').exists(),'Partial attempts preserved; do not blindly rerun'
 with(R/f'fold{f}.log').open('ab',buffering=0)as log:
  c=subprocess.Popen([sys.executable,str(R/'source/experiment.py'),'fold','--fold',str(f)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(dict(status='running',active_fold=f,pid=c.pid,completed=done,original_deadline=D))
  try:code=c.wait(timeout=min(5400,max(1,D-time.time())))
  except subprocess.TimeoutExpired:
   os.killpg(c.pid,signal.SIGTERM)
   try:c.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
   code=124
 if code:write(dict(status='failed_or_partial',fold=f,completed=done,exit_code=code));sys.exit(code)
 done.append(f)
write(dict(status='training_complete',completed=done,original_deadline=D))
with(R/'evaluation.log').open('ab',buffering=0)as log:code=subprocess.call([sys.executable,str(R/'source/point_report.py')],stdout=log,stderr=subprocess.STDOUT)
write(dict(status='complete'if code==0 else'evaluation_failed',completed=done,exit_code=code,original_deadline=D))
