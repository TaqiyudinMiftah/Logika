"""One prespecified average, assessed on aligned reused OOF, with no weight search."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import ks_2samp,rankdata

R=Path(__file__).resolve().parents[1]
plan=json.loads((R/'H005.json').read_text())
for item in plan['reference_inputs']:
    assert hashlib.sha256((R/item['path']).read_bytes()).hexdigest()==item['sha256']
a=np.load(R/'../coverage-transfer-20261021/oof.npz')
b=np.load(R/'ROBUSTNESS_OOF.npz')
assert np.array_equal(a['target'],b['target'])and np.array_equal(a['fold'],b['fold'])
y,fold=b['target'],b['fold']
pred={'V25':a['published_V25'],'B':b['maturity__original']}
pred['equal_mean']=.5*(pred['V25']+pred['B'])
assert np.max(np.abs(a['V25']-a['published_V25']))<2e-6
def score(t,p):
    n=int(t.sum())
    return dict(KS=float(ks_2samp(p[t==0],p[t==1]).statistic),
                AUC=float((rankdata(p)[t==1].sum()-n*(n+1)/2)/(n*(len(t)-n))))
metrics={}
for name,p in pred.items():
    m=score(y,p)
    m['fold_KS']=[score(y[fold==f],p[fold==f])['KS']for f in range(5)]
    metrics[name]=m
effects={}
for name in ['V25','B']:
    diff=np.array(metrics['equal_mean']['fold_KS'])-metrics[name]['fold_KS']
    effects[name]=dict(pooled_delta=metrics['equal_mean']['KS']-metrics[name]['KS'],fold_delta=diff.tolist(),fold_wins=int(sum(diff>0)),worst_fold_delta=float(min(diff)))
accepted=all(e['pooled_delta']>0 and e['fold_wins']>=3 and e['worst_fold_delta']>=-.002 for e in effects.values())
out=dict(status='complete',hypothesis='H005',metrics=metrics,effects=effects,
         promotion_predicate_passed=accepted,independent_confirmation=False,
         public_feedback_used_for_hypothesis=True,no_weight_search=True,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'FIXED_V25_B_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
np.savez_compressed(R/'FIXED_V25_B_OOF.npz',target=y,fold=fold,**pred)
print(json.dumps(out,indent=2))
