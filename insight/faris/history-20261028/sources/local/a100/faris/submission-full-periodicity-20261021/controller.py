"""Finite full-data refit; independent of the research search admission clock."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,json,time,subprocess,sys,traceback
R=Path(__file__).resolve().parent
def write(d):
 p=R/'STATE.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def run(args,log,extra=None):
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.update(extra or{})
 with(R/log).open('ab')as out:subprocess.run([sys.executable,'-u']+args,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
def main():
 assert not(R/'STATE.json').exists();write(dict(status='running',stage='prepare',pid=os.getpid()));run([str(R/'source/prepare.py')],'prepare.log');write(dict(status='running',stage='features',pid=os.getpid()))
 def feature(f):run([str(R/'feature_build/source/build.py'),'--fold',str(f)],f'features_fold{f}.log')
 with ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(feature,range(5)))
 write(dict(status='running',stage='full_data_training',pid=os.getpid(),GPUs=[0,1]))
 def lane(gpu):
  for f in range(gpu,5,2):run([str(R/'source/train.py'),'--fold',str(f)],f'training_fold{f}.log',dict(CUDA_VISIBLE_DEVICES=str(gpu)))
 with ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(lane,[0,1]))
 write(dict(status='running',stage='final_verification',pid=os.getpid()));run([str(R/'source/finish.py')],'verification.log');assert json.loads((R/'FINAL_STATUS.json').read_text())['status']=='passed';write(dict(status='complete',pid=os.getpid(),completed=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:write(dict(status='failed',pid=os.getpid(),error=traceback.format_exc()));raise
