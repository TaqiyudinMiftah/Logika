"""Resumable sharded queue; each bounded fit has a separate memory cgroup."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--lane',type=int,required=True);p.add_argument('--lanes',type=int,default=3);p.add_argument('--pilot',action='store_true');p.add_argument('--start-index',type=int,default=0);p.add_argument('--end-index',type=int);a=p.parse_args()
plan=json.loads((R/'PLAN.json').read_text());catalog=json.loads((R/'CATALOG.json').read_text())
packs=[p for p in catalog['packs'] if not a.pilot or p['id'] in plan['pilot_packs']]
(R/'logs').mkdir(exist_ok=True)
for pi,pack in enumerate(packs):
    if pi<a.start_index or (a.end_index is not None and pi>=a.end_index):continue
    if pi%a.lanes!=a.lane:continue
    for fold in range(5):
        if (R/'STOP').exists() or time.time()>plan['search_deadline_epoch']:sys.exit(0)
        if not a.pilot:
            gate=json.loads((R/'PILOT_GATE.json').read_text());assert gate['status']=='passed_for_declared_development_experiment'
        dest=R/f'runs/small2/{pack["id"]}/fold{fold}'
        if (dest/'RESULT.json').exists():continue
        if dest.exists():continue
        unit=f'logika-horizontal-{a.lane}-{int(time.time())}'
        limit=int(min(900,plan['search_deadline_epoch']-time.time()))
        command=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit',unit,
                 '-p','MemoryMax=18G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
                 '-p',f'WorkingDirectory={R}','-p',f'RuntimeMaxSec={limit}',
                 '--setenv=CUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES','0'),
                 '--setenv=HF_HOME='+os.environ.get('HF_HOME','/mnt/home/faris/logika-model-arena-20261025/storage/huggingface'),
                 '--setenv=LOGIKA_FEATURE_BANK='+os.environ.get('LOGIKA_FEATURE_BANK','/mnt/home/faris/faris/logika-meta-experiments/meta-20261016-v1/bank'),
                 sys.executable,str(R/'source/worker.py'),'--pack',pack['id'],'--fold',str(fold)]
        log=R/'logs'/f'{pack["id"]}_fold{fold}_{int(time.time())}.log'
        start=time.time()
        try:
            with log.open('x') as f:rc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=limit+30).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['systemctl','--user','stop',unit],check=False);rc=124
        event={'pack':pack['id'],'fold':fold,'returncode':rc,'seconds':time.time()-start,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with (R/f'events_lane{a.lane}.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
