"""Owned, finite post-run verification; no training, proposals or new deadline."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,subprocess,sys,hashlib
import numpy as np
from scipy.stats import rankdata
R=Path(__file__).resolve().parent
def write(value,path):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def main():
 deadline=json.loads((R/'training-plan.json').read_text())['original_deadline']
 while time.time()<deadline:
  state=R/'TRAINING_STATUS.json'
  if state.exists():
   status=json.loads(state.read_text())
   if status['status']=='complete':break
   if status['status']in ['failed','partial']:
    write(dict(status='training_not_complete',training=status),R/'FINAL_EVIDENCE.json');return
  if (R/'STOP').exists():return
  time.sleep(10)
 else:
  write(dict(status='deadline_expired',original_deadline=deadline),R/'FINAL_EVIDENCE.json');return
 with (R/'content-audit.log').open('ab')as log:
  subprocess.run([sys.executable,str(R/'audit_completed.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 audit=json.loads((R/'CONTENT_AUDIT.json').read_text());assert audit['completed_folds']==5
 z=np.load(R/'OOF.npz');reported=json.loads((R/'result.json').read_text());y=z['target'].astype(np.int64);fold=z['fold'];assert len(y)==70000 and np.array_equal(np.bincount(fold),np.full(5,14000))
 metrics={}
 for key in ['baseline','maturity_only','coverage_information','replayed_prediction']:
  p=z[key];assert len(p)==70000 and np.isfinite(p).all()and np.all((p>0)&(p<1));order=np.argsort(p,kind='stable');v=p[order];yy=y[order];ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1]
  n=int(y.sum());ks=float(np.max(abs(np.cumsum(yy)[ends]/n-np.cumsum(1-yy)[ends]/(len(y)-n))));auc=float((rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))
  expected=reported['replayed_model_metrics']if key=='replayed_prediction'else reported['metrics'][key];assert abs(ks-expected['KS'])<1e-12 and abs(auc-expected['AUC'])<1e-12;metrics[key]=dict(KS=ks,AUC=auc)
 candidate=reported['metrics']['coverage_information'];base=reported['metrics']['baseline'];fm=reported['fold_metrics'];delta=[fm['coverage_information'][f]['KS']-fm['baseline'][f]['KS']for f in range(5)]
 supported=candidate['KS']>base['KS']and candidate['AUC']>base['AUC']and sum(d>0 for d in delta)>=3 and min(delta)>=-.005
 result=dict(status='passed',metrics=metrics,model_audit_completed_folds=5,independent_metric_method='Tied empirical CDF and rank-sum AUC',provisional_positive_point_evidence=bool(supported),fold_delta_KS=delta,goal_point_reached=candidate['KS']>=.54 and metrics['replayed_prediction']['KS']>=.54,confirmation_opened=False,submission_generated=False,original_deadline=deadline,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Allfive fittedmodel/transformer/context audits andfull70kmetrics verified. Conditional intervals, repeated developmentlabeluse and adaptive featureselection remain limitations. This does not promote a final model or certify private generalization.')
 write(result,R/'FINAL_EVIDENCE.json');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
