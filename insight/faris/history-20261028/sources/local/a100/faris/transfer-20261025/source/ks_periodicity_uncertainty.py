"""Predeclared paired conditional assessment of H008; no further fitting."""
import numpy as np
from scipy.stats import ks_2samp
import baseline as b
from robustness import R,read

result=read(R/'KS_PERIODICITY_RESULT.json')
c=np.load(R/'KS_PERIODICITY_OOF.npz');z=np.load(R/'ROBUSTNESS_OOF.npz')
y,fold=z['target'],z['fold'];candidate=c['two_seed_mean'];control=z['periodicity__two_seed_mean']
assert np.array_equal(c['target'],y)and np.array_equal(c['fold'],fold)
def ks(y,p):
    order=np.argsort(p,kind='stable');t=y[order];s=p[order];ends=np.r_[s[1:]!=s[:-1],True];pos=np.cumsum(t);neg=np.arange(1,len(t)+1)-pos
    return float(np.max(np.abs((neg/(len(t)-t.sum())-pos/t.sum())[ends])))
assert abs(ks(y,candidate)-ks_2samp(candidate[y==1],candidate[y==0]).statistic)<1e-14
groups=[np.flatnonzero((fold==f)&(y==cl))for f in range(5)for cl in(0,1)]
rng=np.random.default_rng(202610258);draws=[]
for _ in range(1000):
    rows=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);draws.append(ks(y[rows],candidate[rows])-ks(y[rows],control[rows]))
interval=np.quantile(draws,[.025,.975]).tolist()
b.dump(dict(status='complete',candidate_KS=ks(y,candidate),control_KS=ks(y,control),point_delta=ks(y,candidate)-ks(y,control),conditional95percent_interval=interval,repetitions=1000,submission_rule_passed=bool(result['point_rule_passed']and interval[0]>0),independent_confirmation=False,limitation='Paired fixed-prediction bootstrap on reused development data; excludes adaptive search and refitting variance.',result_sha256=b.sha(R/'KS_PERIODICITY_RESULT.json'),source_sha256=b.sha(__file__)),R/'KS_PERIODICITY_UNCERTAINTY.json')
np.savez_compressed(R/'KS_PERIODICITY_BOOTSTRAP.npz',delta=draws)
