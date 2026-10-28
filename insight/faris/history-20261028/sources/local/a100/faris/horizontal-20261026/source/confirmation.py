"""Run the inner-selected shortlist with larger checkpoints and eight estimators."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--lane',type=int,required=True);p.add_argument('--lanes',type=int,default=2);a=p.parse_args()
plan=json.loads((R/'PLAN.json').read_text());selection=json.loads((R/'selection/SELECTION.json').read_text())
for i,task in enumerate(selection['tasks']):
    if i%a.lanes!=a.lane:continue
    if time.time()>=plan['deadline_epoch']-1800 or (R/'STOP').exists():break
    dest=R/f'runs/{task["size"]}{task["estimators"]}/{task["pack"]}/fold{task["fold"]}'
    if (dest/'RESULT.json').exists():continue
    if dest.exists():continue
    unit=f'logika-horizontal-confirm-{a.lane}-{int(time.time())}'
    limit=int(min(1800,plan['deadline_epoch']-1800-time.time()))
    cmd=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit',unit,
         '-p','MemoryMax=22G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
         '-p',f'WorkingDirectory={R}','-p',f'RuntimeMaxSec={limit}',
         '--setenv=CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'],
         '--setenv=HF_HOME='+os.environ.get('HF_HOME','/mnt/home/faris/logika-model-arena-20261025/storage/huggingface'),
         '--setenv=LOGIKA_FEATURE_BANK='+os.environ.get('LOGIKA_FEATURE_BANK','/mnt/home/faris/faris/logika-meta-experiments/meta-20261016-v1/bank'),
         sys.executable,str(R/'source/worker.py'),'--pack',task['pack'],'--fold',str(task['fold']),
         '--size',task['size'],'--estimators',str(task['estimators'])]
    log=R/'logs'/f'confirm_{task["size"]}_{task["pack"]}_{task["fold"]}_{int(time.time())}.log'
    start=time.time()
    try:
        with log.open('x') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=limit+30).returncode
    except subprocess.TimeoutExpired:
        subprocess.run(['systemctl','--user','stop',unit],check=False);rc=124
    event={**task,'returncode':rc,'seconds':time.time()-start}
    with (R/f'confirmation_lane{a.lane}.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)
