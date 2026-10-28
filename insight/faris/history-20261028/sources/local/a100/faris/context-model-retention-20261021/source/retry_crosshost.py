from pathlib import Path
import os,time,json,subprocess,sys,signal
R=Path(__file__).resolve().parents[1];D=1792585871.823748
attempt=0
while True:
 assert time.time()<D-10800 and not(R/'STOP').exists()
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 ready=[int(a)for a,b,c in[l.split(',')for l in lines]if int(b)>11000 and int(c)<10]
 reserved=subprocess.check_output(['systemctl','--user','show','logika-kumo-supervised-view-20261021.service','-p','MainPID'],text=True).strip().split('=')[-1]
 if reserved.isdigit()and int(reserved)>0 and Path('/proc/'+reserved).exists():ready=[g for g in ready if g!=0]
 (R/'CROSSHOST_RETRY.json').write_text(json.dumps(dict(status='waiting_for_free_GPU',attempts=attempt,original_deadline=D),indent=2))
 if not ready:time.sleep(10);continue
 attempt+=1;log=R/f'crosshost_retry{attempt}.log'
 with log.open('xb',buffering=0)as f:
  child=subprocess.Popen([sys.executable,str(R/'source/crosshost.py')],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(ready[0])),start_new_session=True)
  (R/'CROSSHOST_RETRY.json').write_text(json.dumps(dict(status='running',pid=child.pid,physical_gpu=ready[0],attempt=attempt,original_deadline=D),indent=2))
  try:code=child.wait(timeout=min(3600,max(1,D-time.time())))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:code=child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
 if code==0:
  assert json.loads((R/'CROSSHOST.json').read_text())['status']=='passed';break
 if attempt>=3 or'OutOfMemoryError'not in log.read_text():
  (R/'CROSSHOST.json').write_text(json.dumps(dict(status='failed',attempt=attempt,exit_code=code)));sys.exit(code or 1)
 time.sleep(10)
