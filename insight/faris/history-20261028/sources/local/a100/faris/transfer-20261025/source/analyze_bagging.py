"""Evaluate only the fixed averages declared in H001 before pilot outcomes."""
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
import baseline as b
from robustness import R, PERIODIC, read

z=np.load(R/'ROBUSTNESS_OOF.npz')
y,fold=z['target'],z['fold']
predictions={}
metrics={}
for recipe in ['baseline','maturity','periodicity']:
    original=z[f'{recipe}__original']
    boot=.3333333333333333*(original+z[f'{recipe}__correction_resample_51025']+z[f'{recipe}__correction_resample_61025'])
    predsets={'original':original,'three_head_mean':boot,'two_seed_mean':z[f'{recipe}__two_seed_mean']}
    for name,p in predsets.items():
        score=b.scores(y,p)
        assert abs(score['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
        score['fold_KS']=[b.scores(y[fold==f],p[fold==f])['KS']for f in range(5)]
        metrics[f'{recipe}__{name}']=score
        predictions[f'{recipe}__{name}']=p
np.savez_compressed(R/'BAGGING_OOF.npz',target=y,fold=fold,**predictions)
b.dump(dict(status='complete',metrics=metrics,hypothesis=read(R/'H001.json'),
            interpretation='Fixed averages; reused development evidence. Seed and bootstrap sensitivity do not create independent validation.'),R/'BAGGING_RESULT.json')
