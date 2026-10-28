from pathlib import Path
import os,sys,time,json,subprocess,fcntl,signal,shutil,traceback
import baseline as b
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def main():
 lock=(R/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=read(R/'training-plan.json');assert p['training_admitted']and read(R/'DATA_READY.json')['status']=='passed'
 assert not(R/'TRAINING_STATUS.json').exists(),'Existing run requires explicit terminal/live-process audit before restart.'
 active={};pending=list(range(5));done=[];failed=[]
 try:
  while pending or active:
   assert time.time()<p['original_deadline']and not(R/'STOP').exists();assert shutil.disk_usage(R).free>5*1024**3
   for gpu,(job,proc,log)in list(active.items()):
    if proc.poll()is None:continue
    log.close();del active[gpu]
    if proc.returncode==0 and(R/f'runs/fold{job}/complete.json').exists():done.append(job)
    else:failed.append(dict(fold=job,returncode=proc.returncode))
   if pending and time.time()<p['original_deadline']-10800:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
     gpu,free,util=[int(s.strip())for s in line.split(',')]
     if gpu not in[0,1]or gpu in active or free<11000 or util>10 or not pending:continue
     job=pending.pop(0);assert not(R/f'runs/fold{job}').exists();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);log=(R/f'fold{job}.log').open('ab');proc=subprocess.Popen([sys.executable,'-u',str(R/'source/worker.py'),'--fold',str(job)],stdout=log,stderr=log,env=env,start_new_session=True);active[gpu]=(job,proc,log)
   b.dump(dict(status='running',completed=done,failed=failed,queued=pending,active=[dict(gpu=g,fold=j,pid=proc.pid)for g,(j,proc,_)in active.items()],original_deadline=p['original_deadline']),R/'TRAINING_STATUS.json')
   if pending and not active and time.time()>=p['original_deadline']-10800:break
   time.sleep(3)
  if len(done)==5:
   b.dump(dict(status='reporting',completed=done,failed=failed,original_deadline=p['original_deadline']),R/'TRAINING_STATUS.json');subprocess.run([sys.executable,str(R/'source/report.py')],check=True,stdout=(R/'evaluation.log').open('ab'),stderr=subprocess.STDOUT);status='complete'
  else:status='partial'
  b.dump(dict(status=status,completed=done,failed=failed,unstarted=pending,original_deadline=p['original_deadline']),R/'TRAINING_STATUS.json')
 except BaseException:
  for job,proc,log in active.values():
   if proc.poll()is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   log.close()
  b.dump(dict(status='failed',error=traceback.format_exc(),completed=done,failed=failed,queued=pending,original_deadline=p['original_deadline']),R/'TRAINING_STATUS.json');raise
if __name__=='__main__':main()
