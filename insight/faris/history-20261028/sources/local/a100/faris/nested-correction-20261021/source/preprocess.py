"""Bounded feature preparation. Never admits a foundation model or reads scores."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,signal,shutil
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];p=json.loads((R/'plan.json').read_text());D=p['original_deadline']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(name,x):
 t=R/(name+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(R/(name+'.json'))
for n,h in p['source_sha256'].items():assert sha(R/'vendor'/n)==h
for n,h in p['preparation_source_sha256'].items():assert sha(R/'source'/n)==h
assert json.loads((R/'ROLE_GATE.json').read_text())['status']=='passed'
assert not p['model_training_admitted']and p['preprocessing_only']
original=pd.read_parquet(Path(p['input'])/'data/train_features.parquet',columns=['application_id','customer_id'])
raw=Path(p['raw']);apps=pd.read_csv(raw/'train_applications.csv',usecols=['customer_id','decision_datetime'],parse_dates=['decision_datetime']);rep=pd.read_csv(raw/'repayment_history.csv',usecols=['customer_id','due_date','max_days_past_due'],parse_dates=['due_date']).merge(apps,on='customer_id',validate='many_to_one');rep=rep[(rep.decision_datetime-rep.due_date).dt.total_seconds()>=86400]
def verify(j):
 d=R/f'fold{j}';c=json.loads((d/'complete.json').read_text());assert sha(d/'features.parquet')==c['feature_table_sha256'];f=pd.read_parquet(d/'features.parquet');ix=np.load(Path(p['input'])/f'artifacts/fold{j}/indices.npz');ci,vi=ix['context'],ix['validation'];job=p['jobs'][str(j)]
 assert c['context_rows']==len(ci)==job['context_rows']and c['validation_rows']==len(vi)==job['query_rows']
 assert len(f)==100000 and f.application_id.is_unique and f.customer_id.is_unique and'target'not in f
 assert f.application_id.iloc[:70000].tolist()==original.application_id.tolist()
 assert hashlib.sha256('\n'.join(original.application_id.iloc[ci]).encode()).hexdigest()==c['fit_ids_sha256']
 fit=set(original.customer_id.iloc[ci]);actual=float(rep[rep.customer_id.isin(fit)].max_days_past_due.ge(1).mean());reported=json.loads((d/'workspace/artifacts/optimization/v4/feature_audit.json').read_text())['mature_late_1_prior'];assert abs(actual-reported)<1e-14
 return dict(job=j,feature_sha256=c['feature_table_sha256'],context_ids_sha256=c['fit_ids_sha256'],mature_late_1_prior=actual)
active={};done=[];checks=[];queue=[0,5]+[j for j in range(30)if j not in [0,5]];failed=False
while queue or active:
 now=time.time()
 for j,a in list(active.items()):
  code=a['process'].poll()
  if code is None and(now>=D or(R/'STOP').exists()):
   os.killpg(a['process'].pid,signal.SIGTERM)
   try:code=a['process'].wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(a['process'].pid,signal.SIGKILL);code=a['process'].wait()
  if code is not None:
   a['log'].close();del active[j]
   if code:failed=True;write('FAILURE',dict(job=j,code=code));continue
   checks.append(verify(j));done.append(j)
 if failed or now>=D-10800 or(R/'STOP').exists():queue=[]
 # Verify both context sizes before admitting the rest of feature construction.
 slots = 2 if not {0, 5} <= set(done) else 4
 while queue and len(active)<slots:
  if not{0,5}<=set(done)and queue[0]not in[0,5]:break
  if shutil.disk_usage(R).free <= 35*1024**3:
   failed=True;queue=[];write('FAILURE',dict(reason='disk headroom below35GiB; active owned builders finish'));break
  j=queue.pop(0);assert not(R/f'fold{j}').exists(),'Partial preprocessing requires explicit review'
  log=(R/f'build{j}.log').open('ab',buffering=0);child=subprocess.Popen([sys.executable,str(R/'source/build.py'),'--fold',str(j)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[j]=dict(process=child,log=log)
 write('PREPROCESSING',dict(status='running',completed=done,active={j:a['process'].pid for j,a in active.items()},queued=queue,original_deadline=D))
 if not queue and not active:break
 time.sleep(5)
write('PREPROCESSING',dict(status='complete'if len(done)==30 else'partial_or_failed',completed=done,checks=checks,original_deadline=D,foundation_model_training_admitted=False))
if len(done)==30:write('INDEPENDENT_AUDIT',dict(status='passed',jobs=30,checks=checks,scope='All output hashes, frozen inner/final membership, independent mature-repayment priors; no evaluation targets. Full vendor source fitting scope inherited from audited snapshots.',raw_snapshot_provenance_certified=False))
