"""Paired, conditional uncertainty for the frozen H006 comparison."""
import os
os.environ['OMP_NUM_THREADS']='1'
import argparse
import numpy as np
import baseline as b
from robustness import R,read

parser=argparse.ArgumentParser()
parser.add_argument('--hypothesis',choices=['H006','H007'],default='H006')
args=parser.parse_args()
tag='CORE_MATURITY' if args.hypothesis=='H006' else 'CORE_LATE30'
assert read(R/f'{tag}_RESULT.json')['status']=='complete'
z=np.load(R/'ROBUSTNESS_OOF.npz')
c=np.load(R/f'{tag}_OOF.npz')
y,fold=z['target'],z['fold']
assert np.array_equal(y,c['target']) and np.array_equal(fold,c['fold'])
candidate=c['two_seed_mean']
comparators={'matched_maturity_two_seed':z['maturity__two_seed_mean'],
             'current_local_best_A':z['periodicity__two_seed_mean']}

def ks(y,p):
    order=np.argsort(p,kind='stable'); t=y[order]; s=p[order]
    ends=np.r_[s[1:]!=s[:-1],True]; pos=np.cumsum(t); neg=np.arange(1,len(t)+1)-pos
    return float(np.max(np.abs((neg/(len(t)-t.sum())-pos/t.sum())[ends])))

groups=[np.flatnonzero((fold==f)&(y==cl)) for f in range(5) for cl in (0,1)]
rng=np.random.default_rng(202610256 if args.hypothesis=='H006' else 202610257)
draws={k:[] for k in comparators}
for _ in range(1000):
    rows=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
    score=ks(y[rows],candidate[rows])
    for key,pred in comparators.items():draws[key].append(score-ks(y[rows],pred[rows]))
result={k:dict(point_delta=ks(y,candidate)-ks(y,p),
               conditional_95pct_interval=np.quantile(draws[k],[.025,.975]).tolist(),
               two_contrast_adjusted_95pct_interval=np.quantile(draws[k],[.0125,.9875]).tolist())
        for k,p in comparators.items()}
b.dump(dict(status='complete',candidate=f'{args.hypothesis}_two_seed_mean',candidate_pooled_KS=ks(y,candidate),
            repetitions=1000,comparisons=result,selection_evidence_sha256=b.sha(R/f'{tag}_RESULT.json'),
            independent_confirmation=False,
            limitation='Conditional on fitted predictions and reused development labels; excludes adaptive search, refitting and public-sample uncertainty.'),R/f'{tag}_UNCERTAINTY.json')
np.savez_compressed(R/f'{tag}_BOOTSTRAP.npz',**draws)
