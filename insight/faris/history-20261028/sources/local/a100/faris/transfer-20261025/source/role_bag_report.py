"""Reconstruct all H010 predictions and assess the frozen comparisons."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
import baseline as b
from robustness import R,A,INPUT,read

D=R/'role_bag'
plan=read(R/'H010.json')
assert b.sha(A/'ranking-pipeline-20261020/assets/classifier.pt')==read(R/'H006.json')['checkpoint_sha256']
train=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','target'])
reference=np.load(R/'ROBUSTNESS_OOF.npz');y,fold=reference['target'],reference['fold']
assert len(y)==70000 and np.array_equal(y,train.target.to_numpy(int))
predictions={f'{scope}_{mode}':np.full(len(y),np.nan)for scope in ['onebag','fivebag']for mode in ['ordinary','direct_KS']}
checks=[]
for outer in range(5):
    shared=D/f'outer{outer}/shared';prep=read(shared/'PREPARED.json')
    frame=pd.read_parquet(shared/'features.parquet');assert frame.application_id.tolist()==train.application_id.tolist()
    ix=np.load(INPUT/f'artifacts/fold{outer}/indices.npz');rows=ix['validation']
    assert prep['preprocessing_ID_hash']==b.ids_hash(train,ix['train']) and prep['preprocessing_fit_rows']==56000
    sp=joblib.load(shared/'spline.joblib');assert sp['fit_rows']==56000 and sp['fit_ids_sha']==b.ids_hash(train,ix['train'])
    spline=b.predict_spline(sp,frame.iloc[rows]);saved_spline=np.load(shared/'spline_validation.npz')
    assert np.max(np.abs(spline-saved_spline['prediction']))<2e-14
    members={m:[]for m in ['ordinary','direct_KS']}
    for member in range(5):
        root=D/f'outer{outer}/member{member}';complete=read(root/'complete.json');assert complete['status']=='complete'
        for n,h in complete['files'].items():assert b.sha(root/n)==h,n
        roles=np.load(D/f'outer{outer}/roles/member{member}.npz')
        assert np.array_equal(roles['validation'],rows)
        assert np.array_equal(np.sort(np.r_[roles['context'],roles['correction']]),np.sort(ix['train']))
        assert not np.intersect1d(rows,np.r_[roles['context'],roles['correction']]).size
        canonical=joblib.load(root/'kumo_model.joblib')
        assert np.array_equal(canonical['context_indices'],roles['context'])
        assert canonical['feature_table_sha256']==b.sha(shared/'features.parquet')
        assert canonical['context_ID_hash']==b.ids_hash(frame,roles['context'])
        assert canonical['target_source_sha256']==b.sha(INPUT/'data/train_features.parquet')
        raw=np.load(root/'raw_predictions.npz')
        assert np.array_equal(raw['rows'],np.r_[roles['correction'],rows])
        assert raw['application_id'].tolist()==frame.application_id.iloc[raw['rows']].tolist()
        for mode in members:
            head=joblib.load(root/f'{mode}_head.joblib')
            pred=.75*b.corrected_probability(head,frame.iloc[rows],raw['prediction'][6000:])[0]+.25*spline
            saved=np.load(root/f'{mode}_validation.npz')
            assert saved['application_id'].tolist()==train.application_id.iloc[rows].tolist()
            assert np.max(np.abs(pred-saved['prediction']))<2e-14
            members[mode].append(saved['prediction'])
            if member==outer:predictions[f'onebag_{mode}'][rows]=saved['prediction']
        checks.append(dict(outer=outer,member=member,roles_verified=True,serialized_models_replayed=True))
    for mode in members:predictions[f'fivebag_{mode}'][rows]=np.mean(members[mode],axis=0)
metrics={}
for name,p in predictions.items():
    assert np.isfinite(p).all()
    s=b.scores(y,p);assert abs(s['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-14
    s['fold_KS']=[b.scores(y[fold==f],p[fold==f])['KS']for f in range(5)];metrics[name]=s
np.savez_compressed(R/'ROLE_BAG_OOF.npz',target=y,fold=fold,**predictions)
b.dump(dict(status='passed',checks=checks,validation_applicants_once=70000,preprocessing_scope='outertraining56k covariates only',heldout_labels_never_fit=True,independent_generalization=False,source_sha256=b.sha(__file__)),R/'ROLE_BAG_OOF_AUDIT.json')

def ks(yy,p):
    order=np.argsort(p,kind='stable');t=yy[order];s=p[order];ends=np.r_[s[1:]!=s[:-1],True];pos=np.cumsum(t);neg=np.arange(1,len(t)+1)-pos
    return float(np.max(np.abs((neg/(len(t)-t.sum())-pos/t.sum())[ends])))

controls={'matched_onebag':predictions['onebag_ordinary'],'previous_B':reference['maturity__original']}
groups=[np.flatnonzero((fold==f)&(y==cl))for f in range(5)for cl in(0,1)]
rng=np.random.default_rng(2026102510);draws={f'{mode}__{control}':[]for mode in ['ordinary','direct_KS']for control in controls}
for _ in range(1000):
    rows=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups])
    base={k:ks(y[rows],p[rows])for k,p in controls.items()}
    for mode in ['ordinary','direct_KS']:
        score=ks(y[rows],predictions[f'fivebag_{mode}'][rows])
        for name in base:draws[f'{mode}__{name}'].append(score-base[name])
comparisons={};eligible=[]
for mode in ['ordinary','direct_KS']:
    p=predictions[f'fivebag_{mode}'];comparisons[mode]={}
    for name,control in controls.items():
        delta=ks(y,p)-ks(y,control)
        interval=np.quantile(draws[f'{mode}__{name}'],[.00625,.99375]).tolist()
        comparisons[mode][name]=dict(KS_delta=delta,four_contrast_adjusted95percent_interval=interval)
    wins=sum(b.scores(y[fold==f],p[fold==f])['KS']>b.scores(y[fold==f],controls['previous_B'][fold==f])['KS']for f in range(5))
    passed=metrics[f'fivebag_{mode}']['KS']>.5314098055476294 and wins>=3 and all(comparisons[mode][name]['four_contrast_adjusted95percent_interval'][0]>0 for name in controls)
    comparisons[mode].update(fold_wins_vs_B=int(wins),promotion_passed=bool(passed))
    if passed:eligible.append(mode)
selected=max(eligible,key=lambda k:metrics[f'fivebag_{k}']['KS'])if eligible else None
b.dump(dict(status='complete',metrics=metrics,comparisons=comparisons,selected=selected,target_054_reached=any(metrics[f'fivebag_{k}']['KS']>=.54 for k in ['ordinary','direct_KS']),hypothesis_sha256=b.sha(R/'H010.json'),independent_confirmation=False,limitation='Reused development labels; bootstrap conditions on the predictions and excludes all adaptive search and refitting variation. Newsharedprior56k scope is distinct from old50k contextpriors.'),R/'ROLE_BAG_RESULT.json')
np.savez_compressed(R/'ROLE_BAG_BOOTSTRAP.npz',**draws)
