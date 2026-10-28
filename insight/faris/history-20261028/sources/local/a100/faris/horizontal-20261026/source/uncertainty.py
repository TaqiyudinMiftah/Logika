"""Paired, class-stratified bootstrap of a frozen OOF prediction comparison."""
from pathlib import Path
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parents[1]
z=np.load(R/'reports/confirmation_OOF.npz');y=z['target'];p=z['confirmed'];base=z['baseline']
positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
def prepare(pred):
    order=np.argsort(pred,kind='stable');values=pred[order]
    ends=np.r_[np.flatnonzero(np.diff(values)!=0),len(values)-1]
    return order,ends
def ks(weights,prepared):
    order,ends=prepared
    signed=np.where(y[order]==1,weights[order]/weights[positive].sum(),-weights[order]/weights[negative].sum())
    return float(np.max(np.abs(np.cumsum(signed)[ends])))
a,b=prepare(p),prepare(base);rng=np.random.default_rng(261026);d=[]
for _ in range(2000):
    sampled=np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)]
    weights=np.bincount(sampled,minlength=len(y));d.append(ks(weights,a)-ks(weights,b))
result={'paired_delta_KS':ks(np.ones(len(y)),a)-ks(np.ones(len(y)),b),
        'bootstrap_percentile_95':np.quantile(d,[.025,.975]).tolist(),'replicates':2000,'seed':261026,
        'unit':'application; resampled separately within target classes',
        'limits':'Conditional reused-development uncertainty. Does not remove adaptive selection bias or establish Kaggle improvement.',
        'prediction_sha256':hashlib.sha256((R/'reports/confirmation_OOF.npz').read_bytes()).hexdigest()}
(R/'reports/UNCERTAINTY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
