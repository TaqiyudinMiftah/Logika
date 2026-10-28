"""Bounded sequential lane; each fold gets its own process and durable status."""
import argparse,datetime,json,os,subprocess,sys,time,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--lane',required=True);a=p.parse_args()
queue=json.loads((R/'QUEUE.json').read_text())[a.lane]
started=time.time();deadline=datetime.datetime.fromisoformat(json.loads((R/'PLAN.json').read_text())['training_deadline_UTC']).timestamp()
registry={x['key']:x for x in json.loads((R/'evidence/registry.json').read_text())}
events=R/'evidence'/f'events_{a.lane}.jsonl'
for key in queue:
    if time.time()>deadline or (R/'STOP').exists():break
    display=registry[key]['display'].replace('/','_').replace(' ','_')
    steps=[(True,0)]+[(False,f)for f in range(5)]
    for smoke,fold in steps:
        path=R/'storage'/('smoke'if smoke else 'runs')/display/'config0'/f'fold{fold}'
        if (path/'RESULT.json').exists():continue
        if path.exists():break
        if time.time()>deadline or (R/'STOP').exists():break
        cmd=[sys.executable,str(R/'source/worker.py'),'--key',key,'--fold',str(fold)]
        if smoke:cmd.append('--smoke')
        limit=int(min(600 if smoke else 3600,max(1,deadline-time.time())))
        unit=f'logika-fit-{a.lane}-{int(time.time())}'
        cmd=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit',unit,
             '-p',f'WorkingDirectory={R}','-p','MemoryMax=22G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
             '-p',f'RuntimeMaxSec={limit}',
             '--setenv=CUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES',''), *cmd]
        logfile=R/'evidence'/f'{a.lane}_{display}_{"smoke"if smoke else "full"}_{fold}.log'
        if logfile.exists():logfile=logfile.with_name(logfile.stem+f'_retry_{int(time.time())}.log')
        start=time.time();status='exited'
        try:
            with logfile.open('x')as out:
                proc=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=min(600 if smoke else 3600,max(1,deadline-time.time())))
            rc=proc.returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['systemctl','--user','stop',unit],check=False)
            rc=124;status='timeout'
        startfile=path/'START.json'
        if startfile.exists():
            scratch=json.loads(startfile.read_text()).get('scratch_path')
            if scratch:
                scratch=Path(scratch).resolve()
                assert scratch.parent==R/'runtime' and scratch.name.startswith('fit-')
                if scratch.exists():shutil.rmtree(scratch)
        event={'key':key,'smoke':smoke,'fold':fold,'status':status,'returncode':rc,'seconds':time.time()-start,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with events.open('a')as out:out.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        if rc:break
print(json.dumps({'lane':a.lane,'status':'lane_finished','seconds':time.time()-started}),flush=True)
