"""Conditional H010 full refit; no run is admitted before frozen selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,time,shutil,subprocess
import numpy as np
import pandas as pd
import joblib
import baseline as b
import feature_head,spline_rolebag
from direct_ks import fit as direct_fit
from maturity_features import MaturityFeatures
from gpu_backend import Engine
from robustness import R,A,INPUT,read

D=R/'candidate_F'
S=Path('/home/faris/logika-transfer-rolebag-20261025/full_cohort')
FULL=A/'submission-full-periodicity-20261021'
PY='/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python'

def check():
    p=read(D/'FULL_REFIT_PLAN.json');assert time.time()<p['training_deadline_epoch']
    assert read(R/'ROLE_BAG_SUBMISSION_ASSESSMENT.json')['eligible_for_research_full_refit'] is True
    assert b.sha(R/'ROLE_BAG_SUBMISSION_ASSESSMENT.json')==p['submission_assessment_sha256']
    assert read(R/'ROLE_BAG_RESULT.json')['selected']==p['selected_mode']
    assert b.sha(R/'ROLE_BAG_RESULT.json')==p['selection_evidence']['sha256']
    for n,h in p['source_sha256'].items():assert b.sha(R/'source'/n)==h,n
    assert b.sha(A/'ranking-pipeline-20261020/assets/classifier.pt')==p['checkpoint_sha256']
    return p

def prepare():
    p=check();assert not S.exists();(S/'source').mkdir(parents=True)
    (S/'vendor').symlink_to(A/'fold-local-priors-20261021/vendor',target_is_directory=True)
    shutil.copy2(R/'source/cohort_fullshare.py',S/'source/build.py')
    inp=S/'input';(inp/'data').mkdir(parents=True)
    (inp/'data/train_features.parquet').symlink_to(INPUT/'data/train_features.parquet')
    shutil.copy2(INPUT/'BASELINE_LOCK.json',inp/'BASELINE_LOCK.json')
    (inp/'artifacts/fold0').mkdir(parents=True)
    np.savez_compressed(inp/'artifacts/fold0/indices.npz',context=np.arange(70000),validation=np.array([],dtype=int))
    plan=read(A/'fold-local-priors-20261021/plan.json')
    plan.update(id='H010-full70kshared-priors',original_deadline=p['training_deadline_epoch'],input=str(inp))
    b.dump(plan,S/'plan.json')
    subprocess.run([PY,str(S/'source/build.py'),'--fold','0'],check=True)
    source=S/'fold0';complete=read(source/'complete.json')
    assert complete['context_rows']==70000 and complete['validation_rows']==0
    assert complete['no_target_labels_available_to_feature_estimators'] is True
    assert b.sha(source/'features.parquet')==complete['feature_table_sha256']
    apps=pd.read_parquet(FULL/'applications.parquet');frame=pd.read_parquet(source/'features.parquet')
    assert len(frame)==100000 and frame.application_id.tolist()==apps.application_id.tolist()and'target'not in frame
    train=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','target'])
    assert complete['fit_ids_sha256']==b.ids_hash(train,np.arange(70000))
    spec=read(R/'role_bag/outer0/shared/PREPARED.json')
    for f in range(5):
        other=read(R/f'role_bag/outer{f}/shared/PREPARED.json')
        for key in ['Kumo_columns','head_columns','spline_columns']:assert other[key]==spec[key]
    cov=pd.read_parquet(FULL/'coverage.parquet').set_index('application_id')
    for c in spec['head_columns']:
        if c in cov:frame[c]=cov.loc[frame.application_id,c].to_numpy()
    history=pd.read_csv(A/'exploration-12h-20261017/raw/repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'])
    factory=MaturityFeatures().fit(apps,history,apps.application_id.iloc[:70000])
    mature=factory.transform(apps,history).set_index('application_id').loc[frame.application_id]
    for c in ['re_late_1_posterior','re_late_30_posterior']:frame[c]=mature[c].to_numpy()
    oldsp=joblib.load(R/'role_bag/outer0/shared/spline.joblib')
    for c in oldsp['categorical']:frame[c]=frame[c].astype(object)
    needed=list(dict.fromkeys(['application_id']+spec['Kumo_columns']+spec['head_columns']+spec['spline_columns']))
    frame=frame[needed].copy();shared=D/'shared';shared.mkdir()
    temp=source/'required100k.parquet';frame.to_parquet(temp,index=False,compression='zstd',compression_level=9)
    size=sum(x.stat().st_size for x in R.rglob('*')if x.is_file()and not x.is_symlink())
    assert size+temp.stat().st_size<read(R/'H010.json')['archive_limit_bytes']
    assert shutil.disk_usage(D).free-temp.stat().st_size>600000000
    shutil.copy2(temp,shared/'features.parquet');shutil.copy2(source/'prior_audit.json',shared/'prior_audit.json');shutil.copy2(source/'complete.json',shared/'COHORT_BUILD_COMPLETE.json')
    factory.save(shared/'maturity_transformer.json')
    training=frame.iloc[:70000].copy();training['target']=train.target.to_numpy(int)
    sp=spline_rolebag.fit(training,spec['spline_columns']);joblib.dump(sp,shared/'spline.joblib',compress=3)
    pred=b.predict_spline(sp,frame.iloc[70000:])
    np.savez_compressed(shared/'spline_test.npz',application_id=frame.application_id.iloc[70000:].to_numpy(str),prediction=pred)
    b.dump(dict(status='passed',prior_fit_rows=70000,prior_fit_ID_hash=b.ids_hash(train,np.arange(70000)),spline_fit_rows=70000,train_rows=70000,test_rows=30000,columns=spec['Kumo_columns'],head_columns=spec['head_columns'],spline_columns=spec['spline_columns'],feature_table_sha256=b.sha(shared/'features.parquet'),test_covariates_fit_shared_priors=False,source_sha256=p['source_sha256']),shared/'PREPARED.json')

def run(member):
    p=check();shared=D/'shared';prep=read(shared/'PREPARED.json')
    assert b.sha(shared/'features.parquet')==prep['feature_table_sha256']
    frame=pd.read_parquet(shared/'features.parquet');train=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','target']);y=train.target.to_numpy(int)
    roles=np.load(FULL/f'input/artifacts/fold{member}/indices.npz')
    assert len(roles['context'])==64000 and len(roles['correction'])==6000
    assert np.array_equal(np.sort(np.r_[roles['context'],roles['correction']]),np.arange(70000))
    assert not np.intersect1d(roles['context'],roles['correction']).size
    root=D/f'final/fold{member}';root.mkdir(parents=True,exist_ok=False);started=time.time()
    context=frame.iloc[roles['context']][['application_id']+prep['columns']].copy();context['target']=y[roles['context']]
    canonical=dict(kind='indexed_canonical_Kumo_context',feature_table=str(shared/'features.parquet'),feature_table_sha256=prep['feature_table_sha256'],context_indices=roles['context'],correction_indices=roles['correction'],target_source=str(INPUT/'data/train_features.parquet'),target_source_sha256=b.sha(INPUT/'data/train_features.parquet'),context_ID_hash=b.ids_hash(frame,roles['context']),columns=prep['columns'],seed=2026+member,num_estimators=8,checkpoint_sha256=p['checkpoint_sha256'])
    joblib.dump(canonical,root/'kumo_model.joblib',compress=3)
    rows=np.r_[roles['correction'],np.arange(70000,100000)];query=frame.iloc[rows][['application_id']+prep['columns']].copy()
    engine=Engine(A/'ranking-pipeline-20261020/assets/classifier.pt',prep['columns'],context)
    raw=engine.fit_predict(np.arange(64000),query,2026+member,cache_path=None)
    assert engine.last_prediction_replay_error==0. and np.isfinite(raw).all();del engine
    np.savez_compressed(root/'raw_predictions.npz',rows=rows,application_id=query.application_id.to_numpy(str),prediction=raw)
    correction=frame.iloc[roles['correction']].copy();correction['target']=y[roles['correction']]
    head=feature_head.fit(correction,raw[:6000],prep['head_columns'],.003,.5)
    if p['selected_mode']=='direct_KS':head=direct_fit(head,correction,raw[:6000])
    joblib.dump(head,root/'head.joblib',compress=3)
    sp=np.load(shared/'spline_test.npz');test=frame.iloc[70000:]
    assert sp['application_id'].tolist()==test.application_id.tolist()==raw['application_id'][6000:].tolist()
    pred=.75*b.corrected_probability(head,test,raw[6000:])[0]+.25*sp['prediction']
    replay=.75*b.corrected_probability(joblib.load(root/'head.joblib'),test,raw[6000:])[0]+.25*sp['prediction']
    assert np.array_equal(pred,replay)
    np.savez_compressed(root/'test_predictions.npz',application_id=test.application_id.to_numpy(str),prediction=pred)
    b.dump(dict(status='complete',member=member,selected_mode=p['selected_mode'],context_rows=64000,correction_rows=6000,shared_prior_rows=70000,spline_rows=70000,seconds=time.time()-started,source_sha256=p['source_sha256'],files={x.name:b.sha(x)for x in root.iterdir()if x.is_file()}),root/'complete.json')
    print(dict(stage='rolebag_full_member_complete',member=member,seconds=time.time()-started),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--member',type=int,choices=range(5));a=p.parse_args()
    prepare()if a.action=='prepare'else run(a.member)
