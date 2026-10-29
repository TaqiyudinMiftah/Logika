"""Finalize the existing collected snapshot after explicit resource withdrawal."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,sys,subprocess,time,hashlib,datetime
R=Path(__file__).resolve().parents[1];C=R/'gpu-extra-views-2002';PY='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
clock=json.load(open(R/'CLOCK.json'));assert (C/'RESOURCE_WITHDRAWAL.json').exists()
for name,args in [('compose',[]),('evaluate',['--final']),('audit',[])]:
 with(C/'logs'/('retirement_'+name+'.log')).open('ab')as f:subprocess.run([PY,str(C/'source'/(name+'.py')),*args],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,clock['deadline_epoch']-time.time()-30))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.load(open(C/'ARTIFACT_CONSISTENCY_AUDIT.json'));assert a['status']=='PASS'
v=dict(stage='complete_resource_withdrawal',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_fits=len(list((C/'runs').glob('*/attempt1/RESULT.json'))),failed_attempts=len(list((C/'runs').glob('*/attempt1/FAILURE.json'))),planned_fits=960,resource_withdrawal_sha256=sha(C/'RESOURCE_WITHDRAWAL.json'),audit_sha256=sha(C/'ARTIFACT_CONSISTENCY_AUDIT.json'),clock_sha256=sha(C/'CLOCK.json'),no_further_training=True,remote_shutdown_state='See NODE2002_SHUTDOWN_STATE.json; offline jobs must not be described as stopped',source_versions_not_merged=True)
for name in ['COMPLETION.json','LIVE_STATE.json']:
 p=C/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
