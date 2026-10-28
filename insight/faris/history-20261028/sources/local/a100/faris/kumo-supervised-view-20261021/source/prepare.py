"""Bounded ordinary preparation service; never reads performance for admission."""
from pathlib import Path
import json,time,subprocess,sys,hashlib
R=Path(__file__).resolve().parents[1];p=json.loads((R/'plan.json').read_text());D=p['original_deadline'];remote='/mnt/home/faris/faris/kumo-supervised-view-20261021'
def state(**kw):
 t=R/'PREPARATION.tmp';t.write_text(json.dumps(dict(original_deadline=D,**kw),indent=2));t.replace(R/'PREPARATION.json')
for n,h in p['source_sha256'].items():assert hashlib.sha256((R/'source'/n).read_bytes()).hexdigest()==h
while True:
 assert time.time()<D-10800 and not(R/'STOP').exists()
 command='for f in 0 1 2 3 4; do test -s '+remote+'/data/fold$f/manifest.json || exit 3; done'
 c=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p2003','faris@gpu-labkc.ub.ac.id',command],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 if c.returncode==0:break
 state(status='waiting_for_all_five_context_only_selectors',last_remote_exit=c.returncode);time.sleep(10)
assert not(R/'data').exists(),'Retain interrupted transfer; explicit review needed'
subprocess.run(['scp','-P2003','-r','faris@gpu-labkc.ub.ac.id:'+remote+'/data',str(R)],check=True)
state(status='data_staged_preflight')
subprocess.run([sys.executable,str(R/'source/experiment.py'),'preflight'],check=True)
assert json.loads((R/'PREFLIGHT.json').read_text())['status']=='passed'
p['model_training_admitted']=True;(R/'plan.json').write_text(json.dumps(p,indent=2));state(status='training_admitted')
code=subprocess.call([sys.executable,str(R/'source/runner.py')]);state(status='complete'if code==0 else'failed_or_partial',exit_code=code);sys.exit(code)
