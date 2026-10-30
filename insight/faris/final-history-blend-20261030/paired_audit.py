import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));from audit_all import interval
P=R/'paired';B=Path('/mnt/home/faris/logika-twelvehour-cross-20261030');j=json.loads((P/'RESULTS.json').read_text());sel=j['selected'];selection=P/'FROZEN_SELECTION.json'
if not selection.exists():selection.write_text(json.dumps(dict(selected=sel,epoch=time.time(),frozen_before_interval=True),indent=2)+'\n')
assert json.loads(selection.read_text())['selected']==sel
y=np.load(B/'evaluation_inputs/target.npy');current=np.load(B/'evaluation_inputs/previous.npz')['prediction'];base=np.load(B/'evaluation_inputs/baseline.npz')['prediction'];fold=np.empty(70000,int)
for f in range(5):fold[np.load(R/f'bank/meta{f}.npz')['rows']]=f
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();records=[]
for c in j['candidates']:
 d=P/'candidates'/c['id'];f=json.loads((d/'FROZEN.json').read_text());assert sha(d/'predictions.npz')==f['prediction_sha256'] and sha(d/'submission_HELD.csv')==f['CSV_sha256']==c['CSV_sha256'];z=np.load(d/'predictions.npz');csv=pd.read_csv(d/'submission_HELD.csv');assert len(set(z['ids']))==30000 and csv.application_id.astype(str).tolist()==z['ids'].astype(str).tolist();assert np.max(abs(csv.default_probability-z['test']))<1e-14;score=float(ks_2samp(z['validation'][y==1],z['validation'][y==0]).statistic);assert abs(score-c['KS'])<1e-12;records.append(c)
p=np.load(P/'candidates'/sel['id']/'predictions.npz')['validation'];fd=[float(ks_2samp(p[(fold==f)&(y==1)],p[(fold==f)&(y==0)]).statistic-ks_2samp(current[(fold==f)&(y==1)],current[(fold==f)&(y==0)]).statistic)for f in range(5)]
v=dict(status='PASS',selected=sel,candidates_replayed=len(records),prediction_hash_row_CSV_SciPy_replay='PASS',fold_delta_current=fd,conditional_interval_current=interval(y,p,current,fold),conditional_interval_baseline=interval(y,p,base,fold),all70000_adaptively_reused=True,untouched_audit=False,development_target054=sel['KS']>=.54,new_Kaggle_submission=False,new_base_refits=False,warning='ConditionalfittedpredictionCI excludes allselectionadaptation/refitting andpublicprivatetransfer. Savedproductionmodelpredictions are paired with prior validation recipes; testKSunknown.')
v['supported_new_gain']=sel['delta_current']>=.0005 and v['conditional_interval_current']['percentile95'][0]>0 and sum(x>0 for x in fd)>=4
(P/'FINAL_AUDIT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
