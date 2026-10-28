"""Finalist B: maturity-only, established seed policy, complete 70k training graph."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,time
import numpy as np
import pandas as pd
import joblib
import baseline as b
import feature_head,spline_full
from maturity_features import MaturityFeatures
from robustness import R,A,INPUT,MATURITY,REPLACEMENT,read

FULL=A/'submission-full-periodicity-20261021'
DEST=R/'candidate_B'

def run(f):
    plan=read(DEST/'FULL_REFIT_PLAN.json')
    assert time.time()<plan['training_deadline_epoch']
    for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h,n
    assert b.sha(R/'ROBUSTNESS_RESULT.json')==plan['selection_evidence']['sha256']
    root=DEST/f'final/fold{f}';root.mkdir(parents=True,exist_ok=False)
    old=FULL/f'models/fold{f}';roles=np.load(FULL/f'input/artifacts/fold{f}/indices.npz')
    features=FULL/f'feature_build/fold{f}'
    assert b.sha(features/'features.parquet')==read(features/'complete.json')['feature_table_sha256']
    frame=pd.read_parquet(features/'features.parquet');apps=pd.read_parquet(FULL/'applications.parquet')
    assert frame.application_id.tolist()==apps.application_id.tolist()and'target'not in frame
    y=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['target']).target.to_numpy(int)
    raw=np.load(old/'raw_predictions.npz')
    assert np.array_equal(raw['rows'],np.r_[roles['correction'],np.arange(70000,100000)])
    cov=pd.read_parquet(FULL/'coverage.parquet').set_index('application_id').loc[frame.application_id]
    headspec=joblib.load(MATURITY/f'heads/matched_maturity/fold{f}/model.joblib')
    spspec=joblib.load(REPLACEMENT/f'fold{f}/spline.joblib')
    for c in headspec['columns']:
        if c in cov:frame[c]=cov[c].to_numpy()
    history=pd.read_csv(A/'exploration-12h-20261017/raw/repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'])
    transformer=MaturityFeatures.load(old/'maturity_transformer.json')
    mature=transformer.transform(apps,history).set_index('application_id').loc[frame.application_id]
    for c in ['re_late_1_posterior','re_late_30_posterior']:frame[c]=mature[c].to_numpy()
    transformer.save(root/'maturity_transformer.json')
    for c in spspec['categorical']:frame[c]=frame[c].astype(object)
    correction=frame.iloc[roles['correction']].copy();correction['target']=y[roles['correction']]
    head=feature_head.fit(correction,raw['prediction'][:6000],headspec['columns'],.003,.5)
    training=frame.iloc[:70000].copy();training['target']=y
    sp=spline_full.fit(training,spspec['columns'])
    joblib.dump(head,root/'head_original.joblib',compress=3);joblib.dump(sp,root/'spline.joblib',compress=3)
    test=frame.iloc[70000:].copy();assert raw['application_id'][6000:].tolist()==test.application_id.tolist()
    pred=.75*b.corrected_probability(head,test,raw['prediction'][6000:])[0]+.25*b.predict_spline(sp,test)
    assert np.isfinite(pred).all()and((pred>0)&(pred<1)).all()
    np.savez_compressed(root/'test_predictions.npz',application_id=test.application_id.to_numpy(str),prediction=pred)
    test[['application_id']+list(dict.fromkeys(headspec['columns']+spspec['columns']))].to_parquet(root/'test_features.parquet',index=False)
    b.dump(dict(status='complete',fold=f,variant='maturity',averaging='single_seed',context_rows=64000,correction_rows=6000,spline_rows=70000,
                canonical_context_source=str(old/'kumo_model.joblib'),canonical_context_sha256=b.sha(old/'kumo_model.joblib'),
                original_raw_source=str(old/'raw_predictions.npz'),original_raw_sha256=b.sha(old/'raw_predictions.npz'),
                bundles=[dict(head_file='head_original.joblib',raw_mode='original',draw='ordinary')],
                source_sha256=plan['source_sha256'],files={p.name:b.sha(p)for p in root.iterdir()if p.is_file()}),root/'complete.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);args=p.parse_args();run(args.fold)
