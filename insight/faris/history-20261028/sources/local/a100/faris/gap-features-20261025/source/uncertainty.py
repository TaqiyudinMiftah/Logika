"""Paired uncertainty for the predeclared feature families; no selection or refitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/home/logika-faris-exploration/transfer-20261025')
sys.path.insert(0,str(OLD/'source'))
from joint_correction import weighted_ks
import baseline as b
start=time.time(); z=np.load(R/'OOF.npz'); ref=np.load(OLD/'ROBUSTNESS_OOF.npz')
y=z['target']; baseline=ref['maturity__two_seed_mean']
families=['liquidity','alignment','application_path','alignment_null']
pred={'baseline':baseline,**{c:z[f'{c}__mean']for c in families}}
orders={k:np.argsort(p,kind='stable')for k,p in pred.items()}
ends={k:np.r_[p[orders[k]][1:]!=p[orders[k]][:-1],True]for k,p in pred.items()}
for k,p in pred.items(): assert abs(weighted_ks(y[orders[k]],np.ones(len(y)),ends[k])-b.scores(y,p)['KS'])<1e-12
rng=np.random.default_rng(251025); groups=[np.flatnonzero(y==c)for c in [0,1]];samples={k:[]for k in pred}
for _ in range(1000):
    counts=np.bincount(np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]),minlength=len(y))
    for k in pred:samples[k].append(weighted_ks(y[orders[k]],counts[orders[k]],ends[k]))
result={}
for c in families:
    d=np.array(samples[c])-samples['baseline']
    result[c]={'delta_KS':b.scores(y,pred[c])['KS']-b.scores(y,baseline)['KS'],
        'paired_95_CI':np.quantile(d,[.025,.975]).tolist(),
        'three_family_adjusted_CI':np.quantile(d,[.05/6,1-.05/6]).tolist(),
        'bootstrap_probability_positive':float(np.mean(d>0))}
d=np.array(samples['alignment'])-samples['alignment_null']
result['alignment_minus_permuted']={'delta_KS':b.scores(y,pred['alignment'])['KS']-b.scores(y,pred['alignment_null'])['KS'],'paired_95_CI':np.quantile(d,[.025,.975]).tolist()}
out={'status':'complete','draws':1000,'seed':251025,'conditional_reused_development':True,'independent_confirmation':False,
     'conclusion':'No family passes predeclared promotion rule; all are negative on both Kumo seeds. No new submission.',
     'seconds':time.time()-start,'contrasts':result}
(R/'UNCERTAINTY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
ledger=json.loads((R/'COMPUTE_LEDGER.json').read_text());ledger['uncertainty_seconds']=out['seconds'];ledger['union_elapsed_compute_seconds']+=out['seconds']
(R/'COMPUTE_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
