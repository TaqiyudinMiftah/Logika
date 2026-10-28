"""Paired conditional bootstrap for frozen finalist A; no model selection."""
import os
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import baseline as b
from robustness import R, read

def ks(y,p):
    order=np.argsort(p,kind='stable')
    t=y[order]
    s=p[order]
    ends=np.r_[s[1:]!=s[:-1],True]
    pos=np.cumsum(t)
    neg=np.arange(1,len(t)+1)-pos
    return float(np.max(np.abs((neg/(len(t)-t.sum())-pos/t.sum())[ends])))

z=np.load(R/'ROBUSTNESS_OOF.npz')
y,fold=z['target'],z['fold']
candidate=z['periodicity__two_seed_mean']
comparators={'original_periodicity':z['periodicity__original'], 'repaired_V25':z['baseline__original']}
groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in (0,1)]
rng=np.random.default_rng(202610251)
draws={k:[]for k in comparators}
for i in range(1000):
    rows=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups])
    score=ks(y[rows],candidate[rows])
    for k,p in comparators.items():draws[k].append(score-ks(y[rows],p[rows]))
result={k:{'point_delta':ks(y,candidate)-ks(y,p),'conditional_95pct_interval':np.quantile(draws[k],[.025,.975]).tolist(),'two_contrast_adjusted_95pct_interval':np.quantile(draws[k],[.0125,.9875]).tolist()}for k,p in comparators.items()}
b.dump(dict(status='complete',candidate='periodicity__two_seed_mean',candidate_pooled_KS=ks(y,candidate),comparisons=result,repetitions=1000,
            limitation='Conditional on fitted predictions and reused development labels. Excludes refitting variance, all earlier search, and Kaggle sampling uncertainty. No independent confidence claim.'),R/'FINALIST_A_UNCERTAINTY.json')
np.savez_compressed(R/'FINALIST_A_BOOTSTRAP.npz',**draws)
