"""Owned-node member queue; each complete member is a compact checkpoint."""
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--kind',default='cann');p.add_argument('--owner',type=int,default=0);p.add_argument('--owners',type=int,default=1);p.add_argument('--slots',type=int,default=2);a=p.parse_args()
def one(task):
 seed,fold=task;d=R/f'predictions/{a.kind}/s{seed}/fold{fold}'
 if d.exists():return
 if time.time()>=json.load(open(R/'CLOCK.json'))['admission_end_epoch']:return
 logs=R/'logs';logs.mkdir(exist_ok=True)
 with(logs/f'{a.kind}_{seed}_{fold}.log').open('ab')as f:
  q=subprocess.run([sys.executable,'-u',str(R/'source/refit.py'),'--kind',a.kind,'--seed',str(seed),'--fold',str(fold)],stdout=f,stderr=subprocess.STDOUT,timeout=1900,env={**os.environ,**({'CUDA_VISIBLE_DEVICES':''}if a.kind=='cann'else{})})
 return dict(seed=seed,fold=fold,returncode=q.returncode)
tasks=[(s,f)for s in [2026,7731]for f in range(5)];mine=[t for i,t in enumerate(tasks)if i%a.owners==a.owner]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.slots)as ex:
 for v in ex.map(one,mine):print(json.dumps(v),flush=True)
(R/f'QUEUE_FINAL_{a.kind}_{a.owner}.json').write_text(json.dumps(dict(epoch=time.time(),tasks=mine))+'\n')
