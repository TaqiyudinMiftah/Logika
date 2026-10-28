"""Independent applicant alignment and official-metric readback; no fitting."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
import baseline as b
from robustness import R,INPUT,read

parser=argparse.ArgumentParser()
parser.add_argument('--hypothesis',choices=['H006','H007'],required=True)
args=parser.parse_args()
tag,directory=('CORE_MATURITY','core_maturity') if args.hypothesis=='H006' else ('CORE_LATE30','core_late30')
result=read(R/f'{tag}_RESULT.json')
plan=read(R/f'{args.hypothesis}.json')
reference=np.load(R/'ROBUSTNESS_OOF.npz')
saved=np.load(R/f'{tag}_OOF.npz')
train=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','target'])
y,fold=reference['target'],reference['fold']
assert len(train)==70000 and train.application_id.is_unique
assert np.array_equal(y,train.target.to_numpy())
assert np.array_equal(saved['target'],y) and np.array_equal(saved['fold'],fold)
predictions={}
for mode in ['original','fresh']:
    pred=np.full(len(y),np.nan)
    covered=[]
    for f in range(5):
        root=R/directory/f'fold{f}'
        complete=read(root/'complete.json')
        assert complete['status']=='complete' and complete['source_sha256']==plan['source_sha256']
        p=np.load(root/f'{mode}_validation.npz')
        rows=np.flatnonzero(fold==f)
        assert len(rows)==14000 and np.array_equal(p['rows'],rows)
        assert p['application_id'].tolist()==train.application_id.iloc[rows].tolist()
        assert np.isfinite(p['prediction']).all()
        pred[rows]=p['prediction'];covered.extend(rows.tolist())
    assert len(set(covered))==70000 and len(covered)==70000
    assert np.array_equal(pred,saved[mode])
    predictions[mode]=pred
predictions['two_seed_mean']=.5*(predictions['original']+predictions['fresh'])
assert np.array_equal(predictions['two_seed_mean'],saved['two_seed_mean'])
checks={}
reference_keys={'original':'maturity__original','fresh':'maturity__fresh_kumo_seed','two_seed_mean':'maturity__two_seed_mean'}
for mode,p in predictions.items():
    score=float(ks_2samp(p[y==1],p[y==0]).statistic)
    auc=float(roc_auc_score(y,p))
    assert abs(score-result['metrics'][mode]['candidate']['KS'])<1e-14
    assert abs(auc-result['metrics'][mode]['candidate']['AUC'])<1e-14
    control=reference[reference_keys[mode]]
    delta=score-float(ks_2samp(control[y==1],control[y==0]).statistic)
    assert abs(delta-result['effects'][mode]['KS_delta'])<1e-14
    differences=[]
    for f in range(5):
        rows=fold==f;yy=y[rows]
        differences.append(float(ks_2samp(p[rows][yy==1],p[rows][yy==0]).statistic-ks_2samp(control[rows][yy==1],control[rows][yy==0]).statistic))
    assert np.max(np.abs(np.array(differences)-result['effects'][mode]['fold_KS_delta']))<1e-14
    checks[mode]=dict(KS=score,AUC=auc,KS_delta=delta,fold_wins=int(sum(d>0 for d in differences)))
accepted=all(checks[m]['KS_delta']>0 for m in checks) and checks['two_seed_mean']['KS']>plan['current_best_local_KS'] and checks['two_seed_mean']['fold_wins']>=3
assert bool(accepted)==result['promotion_predicate_passed']
b.dump(dict(status='passed',applicants=70000,coverage_once=True,ID_and_target_alignment=True,
            metrics=checks,promotion_rule_recomputed=True,promotion_passed=bool(accepted),
            official_scipy_metric=True,independent_generalization_certified=False,
            result_sha256=b.sha(R/f'{tag}_RESULT.json'),source_sha256=b.sha(__file__)),R/f'{tag}_OOF_AUDIT.json')
