"""Independently replay selected H010 heads/spline and verify full-role lineage."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import baseline as b
from robustness import R,A,INPUT,read
D=R/'candidate_F';plan=read(D/'FULL_REFIT_PLAN.json');shared=D/'shared'
assert read(R/'ROLE_BAG_SUBMISSION_ASSESSMENT.json')['eligible_for_research_full_refit'] is True
assert b.sha(R/'ROLE_BAG_SUBMISSION_ASSESSMENT.json')==plan['submission_assessment_sha256']
assert read(R/'ROLE_BAG_RESULT.json')['selected']==plan['selected_mode']
assert b.sha(R/'ROLE_BAG_RESULT.json')==plan['selection_evidence']['sha256']
prep=read(shared/'PREPARED.json')
assert b.sha(shared/'features.parquet')==prep['feature_table_sha256']
frame=pd.read_parquet(shared/'features.parquet');train=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','target'])
assert len(frame)==100000 and frame.application_id.is_unique
assert frame.application_id.iloc[:70000].tolist()==train.application_id.tolist()
assert not set(frame.application_id.iloc[70000:])&set(train.application_id)
all_ids=b.ids_hash(train,np.arange(70000));cohort=read(shared/'COHORT_BUILD_COMPLETE.json')
assert cohort['context_rows']==70000 and cohort['fit_ids_sha256']==all_ids and cohort['validation_rows']==0
assert cohort['no_target_labels_available_to_feature_estimators'] is True
mature=read(shared/'maturity_transformer.json')
assert mature['fit_rows']==70000 and mature['fit_application_ids_sha256']==all_ids and mature['default_labels_used'] is False
sp=joblib.load(shared/'spline.joblib');assert sp['fit_rows']==70000 and sp['fit_ids_sha']==all_ids
assert len(sp['columns'])==77 and sp['columns']==prep['spline_columns']
spline=b.predict_spline(sp,frame.iloc[70000:]);saved_sp=np.load(shared/'spline_test.npz')
assert np.array_equal(spline,saved_sp['prediction'])
sample=pd.read_csv(A/'exploration-12h-20261017/raw/sample_submission(3).csv')
parts=[];checks=[]
for f in range(5):
    root=D/f'final/fold{f}';complete=read(root/'complete.json');assert complete['status']=='complete'
    assert complete['source_sha256']==plan['source_sha256']
    assert complete['selected_mode']==plan['selected_mode']
    for n,h in complete['files'].items():assert b.sha(root/n)==h
    c=joblib.load(root/'kumo_model.joblib');ctx,corr=c['context_indices'],c['correction_indices']
    assert len(ctx)==64000 and len(corr)==6000 and not np.intersect1d(ctx,corr).size
    assert np.array_equal(np.sort(np.r_[ctx,corr]),np.arange(70000))
    assert c['feature_table_sha256']==prep['feature_table_sha256']
    assert c['target_source_sha256']==b.sha(INPUT/'data/train_features.parquet')
    assert c['context_ID_hash']==b.ids_hash(frame,ctx)
    assert c['seed']==2026+f and c['num_estimators']==8 and c['checkpoint_sha256']==plan['checkpoint_sha256']
    raw=np.load(root/'raw_predictions.npz');rows=np.r_[corr,np.arange(70000,100000)]
    assert np.array_equal(raw['rows'],rows)and raw['application_id'].tolist()==frame.application_id.iloc[rows].tolist()
    head=joblib.load(root/'head.joblib');assert head['columns']==prep['head_columns'] and len(head['columns'])==18
    assert head['spec']=={'correction_strength':.5,'penalty':.003}
    if plan['selected_mode']=='direct_KS':
        assert head['direct_KS']['temperature']==.5 and head['direct_KS']['trust']==1. and head['direct_KS']['ridge']==.003
    else:assert 'direct_KS' not in head
    prediction=.75*b.corrected_probability(head,frame.iloc[70000:],raw['prediction'][6000:])[0]+.25*spline
    saved=np.load(root/'test_predictions.npz');assert saved['application_id'].tolist()==frame.application_id.iloc[70000:].tolist()
    assert np.array_equal(prediction,saved['prediction'])
    parts.append(pd.Series(prediction,index=frame.application_id.iloc[70000:]).reindex(sample.application_id).to_numpy())
    checks.append(dict(member=f,context_labels_and_roles_verified=True,model_replay_exact=True,training_union_70000=True,shared_prior_rows=70000))
csv=D/plan['submission_filename'];actual=pd.read_csv(csv)
assert actual.columns.tolist()==['application_id','default_probability']and len(actual)==30000
assert actual.application_id.tolist()==sample.application_id.tolist()
assert np.isfinite(actual.default_probability).all()and actual.default_probability.between(0,1,inclusive='neither').all()
error=float(np.max(np.abs(actual.default_probability.to_numpy()-np.mean(parts,axis=0))));assert error<1e-15
b.dump(dict(status='passed',checks=checks,CSV_rows=30000,sample_ID_order_verified=True,max_mean_prediction_error=error,submission_sha256=b.sha(csv),no_new_model_fit=True,independent_generalization_certified=False,verification_source_sha256=b.sha(__file__)),D/'FINAL_INDEPENDENT_AUDIT.json')
