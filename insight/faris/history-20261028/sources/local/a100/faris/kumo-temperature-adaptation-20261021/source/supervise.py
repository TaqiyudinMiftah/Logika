"""Owned subprocess deadline enforcement and one terminal evaluator."""
import os,signal,time,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];p=json.loads((R/'plan.json').read_text());deadline=p['original_deadline']
def write(x):
 t=R/'SUPERVISOR.tmp';t.write_text(json.dumps(x,indent=2));t.replace(R/'SUPERVISOR.json')
with (R/'training.log').open('a',buffering=1)as log:
 child=subprocess.Popen([sys.executable,str(R/'source/run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(dict(status='running',pid=child.pid,supervisor_pid=os.getpid(),started=time.time(),original_deadline=deadline))
 try:code=child.wait(timeout=max(1,deadline-time.time()))
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  write(dict(status='partial',reason='original deadline',pid=child.pid));sys.exit(1)
 write(dict(status='training_complete'if code==0 else'failed',exit_code=code,pid=child.pid,finished=time.time()))
 if code:sys.exit(code)
 with (R/'evaluation.log').open('a',buffering=1)as elog:
  child=subprocess.Popen([sys.executable,str(R/'source/evaluate.py')],stdout=elog,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=child.wait(timeout=max(1,deadline-time.time()))
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM);child.wait();code=124
 write(dict(status='complete'if code==0 else'evaluation_failed',exit_code=code,finished=time.time(),original_deadline=deadline))
