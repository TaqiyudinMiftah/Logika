"""Collect/score only completed predeclared Helma designs; never trains on evaluator host."""
from pathlib import Path
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='2'
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];P=R.parent;sys.path.insert(0,str(R/'source'))
from common import load,dump,sha,utc,tasks,check_seal
PY='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
def module(name,args=()):
 with(R/'logs'/f'{name}.log').open('ab')as f:subprocess.run([PY,str(R/'source'/f'{name}.py'),*args],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,int(load('CLOCK.json')['deadline_epoch']-time.time()-10)))
try:
 check_seal();last=0;clock=load('CLOCK.json')
 while time.time()<clock['model_deadline_epoch']-90:
  count=len(list((R/'runs').glob('*/attempt1/RESULT.json')))
  if time.time()-last>60:module('compose');module('evaluate');last=time.time()
  dump(dict(stage='running',UTC=utc(),completed_fits=count,failed_attempts=len(list((R/'runs').glob('*/attempt1/FAILURE.json'))),planned_fits=len(tasks()),owner='helma_account2',original_deadline_epoch=clock['deadline_epoch']),R/'LIVE_STATE.json')
  if(R/'BACKEND_COLLECTION_account2_COMPLETE.json').exists():break
  time.sleep(15)
 # Controller has its own original-clock bounded final collection and owned release.
 while not(R/'BACKEND_COLLECTION_account2_COMPLETE.json').exists() and time.time()<clock['model_deadline_epoch']+90:time.sleep(10)
 assert(R/'BACKEND_COLLECTION_account2_COMPLETE.json').exists(),'Final collection/release must be reconciled before terminal selection'
 module('compose');module('evaluate',['--final']);module('audit')
 v=dict(stage='complete',UTC=utc(),completed_fits=len(list((R/'runs').glob('*/attempt1/RESULT.json'))),failed_attempts=len(list((R/'runs').glob('*/attempt1/FAILURE.json'))),planned_fits=len(tasks()),audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json'),original_deadline_epoch=clock['deadline_epoch']);dump(v,R/'COMPLETION.json');dump(v,R/'LIVE_STATE.json')
except BaseException:
 p=R/'private/COORDINATOR_FAILURE.log';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc());dump(dict(status='execution_review_required',UTC=utc(),trace_sha256=sha(p)),R/'COORDINATOR_FAILURE.json');raise
