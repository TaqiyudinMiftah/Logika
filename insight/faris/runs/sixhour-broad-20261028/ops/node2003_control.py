"""LAN-only CPU spillover for folds3/4; no target arrays, no foreign workloads."""
from pathlib import Path
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import check_seal,dump,utc
C=check_seal();H='faris@gpu-labkc.ub.ac.id';SSH=['ssh','-oBatchMode=yes','-oConnectTimeout=15','-p2003',H];transport='ssh -oBatchMode=yes -oConnectTimeout=15 -p2003';T='/mnt/home/faris/logika-sixhour-broad-20261028';PYTHON=T+'/env311/bin/python'
def run(args,timeout=900):return subprocess.run(args,text=True,capture_output=True,check=True,timeout=timeout)
try:
 while not(R/'PREPARATION_AUDIT.json').exists():
  assert time.time()<C['admission_end_epoch'];time.sleep(15)
 # Only fitting packets and sealed definitions; evaluator targets stay on2001.
 run(['rsync','-az','-e',transport,'--exclude=private','--exclude=env311','--exclude=packets','--exclude=evaluation_inputs','--exclude=runs','--exclude=logs','--exclude=*.tar.gz','--exclude=__pycache__',str(R)+'/',H+':'+T+'/'],timeout=300)
 for f in [3,4]:
  run(SSH+['mkdir -p '+T+'/packets/fold'+str(f)])
  run(['rsync','-az','-e',transport,str(R/f'packets/fold{f}')+'/',H+':'+T+'/packets/fold'+str(f)+'/'],timeout=600)
 setup='/mnt/home/faris/faris/kumo-coverage-placement-20261021/.venv/bin/python -m venv --system-site-packages '+T+'/env311 && '+PYTHON+' -m pip -q install numpy==2.2.6 pandas==2.3.3 pyarrow==24.0.0 scipy==1.16.3 scikit-learn==1.7.2 lightgbm==4.6.0 catboost==1.2.8 xgboost==3.2.0 > '+T+'/environment_setup.log 2>&1'
 run(SSH+[setup],timeout=900)
 while not(R/'GATE.json').exists():assert time.time()<C['admission_end_epoch'];time.sleep(10)
 run(['rsync','-az','-e',transport,'--include=GATE*.json','--exclude=*',str(R)+'/',H+':'+T+'/'])
 run(SSH+['systemd-run --user --unit=logika-sixhour-cpu-20261028 --property=MemoryMax=20G --property=CPUQuota=400% --property=RuntimeMaxSec=21600 --working-directory='+T+' /bin/sh -c '+"'exec "+PYTHON+' -u ops/worker_queue.py --resource cpu --slots 2 --folds 3,4 --name CPU2003_QUEUE > CPU2003_QUEUE.log 2>&1'+"'"])
 while time.time()<C['model_deadline_epoch']+30:
  run(['rsync','-az','-e',transport,'--include=GATE*.json','--exclude=*',str(R)+'/',H+':'+T+'/'])
  mirror=R/'private/node2003';mirror.mkdir(parents=True,exist_ok=True)
  run(['rsync','-az','-e',transport,'--include=runs/***','--include=dispatch/***','--include=CPU2003_QUEUE*.json','--exclude=*',H+':'+T+'/',str(mirror)+'/'],timeout=180)
  run(['rsync','-az',str(mirror)+'/',str(R)+'/'],timeout=60)
  dump(dict(stage='CPU_spillover_active',UTC=utc(),source='node2003',folds=[3,4]),R/'NODE2003_EXECUTION.json')
  if(mirror/'CPU2003_QUEUE_FINAL.json').exists():break
  time.sleep(25)
 dump(dict(stage='collection_complete',UTC=utc()),R/'NODE2003_COLLECTION_COMPLETE.json')
except BaseException:
 (R/'logs/NODE2003_CONTROL_FAILURE.txt').write_text(traceback.format_exc());dump(dict(stage='execution_failure',UTC=utc(),diagnostic='logs/NODE2003_CONTROL_FAILURE.txt'),R/'NODE2003_EXECUTION.json');raise
