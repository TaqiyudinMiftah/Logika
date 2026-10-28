"""Conditional score-precision illustration, not hidden-label inference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
import baseline as b
from joint_correction import weighted_ks

R=Path(__file__).resolve().parents[1]
plan=json.loads((R/'SCORE_SAMPLING_PLAN.json').read_text())
assert b.sha(__file__)==plan['source_sha256']
z=np.load(R/'ROBUSTNESS_OOF.npz');y=z['target']
predictions={'A':z['periodicity__two_seed_mean'],'B':z['maturity__original']}
info={}
for name,p in predictions.items():
    order=np.argsort(p,kind='stable');s=p[order]
    info[name]=(order,y[order],np.r_[s[1:]!=s[:-1],True])
rng=np.random.default_rng(2026102516)
records=[];saved={}
for n in plan['illustrative_sample_sizes']:
    draws={name:[]for name in predictions}
    for _ in range(plan['repetitions']):
        counts=rng.multinomial(n,np.full(len(y),1/len(y)))
        for name,(order,labels,ends)in info.items():
            draws[name].append(weighted_ks(labels,counts[order],ends))
    a,c=np.array(draws['A']),np.array(draws['B']);delta=a-c
    records.append(dict(illustrative_n=n,
                        A_interval95=np.quantile(a,[.025,.975]).tolist(),
                        B_interval95=np.quantile(c,[.025,.975]).tolist(),
                        A_minus_B_interval95=np.quantile(delta,[.025,.975]).tolist(),
                        fraction_B_exceeds_A=float(np.mean(c>a)),
                        B_median=float(np.median(c)),
                        B_fraction_at_or_above_observed_public=float(np.mean(c>=.56818))))
    for name,value in draws.items():saved[f'{name}_n{n}']=value
out=dict(status='complete',original_scores={name:b.scores(y,p)for name,p in predictions.items()},
         illustrations=records,actual_public_sample_size_known=False,actual_public_labels_used=False,
         data_scope='Previously exposed70000 originalOOF observations and fixedpredictions; resampling with replacement.',
         limitations=['These are conditional sampling illustrations, not confidence intervals for actual Kaggle scores.',
                     'No adjustment for adaptive model selection, refitting, different test features or conditional target shift.',
                     'No resampled score is a newly achieved development result. No candidate selection rule changes.'],
         source_sha256=b.sha(__file__),plan_sha256=b.sha(R/'SCORE_SAMPLING_PLAN.json'))
assert not(R/'SCORE_SAMPLING_RESULT.json').exists()
np.savez_compressed(R/'SCORE_SAMPLING_DRAWS.npz',**saved)
b.dump(out,R/'SCORE_SAMPLING_RESULT.json')
print(json.dumps(out,indent=2),flush=True)
