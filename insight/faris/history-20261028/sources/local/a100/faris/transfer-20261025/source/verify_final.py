"""Independent arithmetic and role audit of saved finalist models; no fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import baseline as b
from robustness import R,A,INPUT,read

R=R/os.environ.get('CANDIDATE_DIR','.')

plan=read(R/'FULL_REFIT_PLAN.json')
sample=pd.read_csv(A/'exploration-12h-20261017/raw/sample_submission(3).csv')
train=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','target'])
full=A/'submission-full-periodicity-20261021'
predictions=[]
checks=[]
for f in range(5):
    root=R/f'final/fold{f}'
    complete=read(root/'complete.json')
    for name,digest in complete['files'].items():assert b.sha(root/name)==digest,name
    assert b.sha(Path(complete['canonical_context_source']))==complete['canonical_context_sha256']
    assert b.sha(Path(complete['original_raw_source']))==complete['original_raw_sha256']
    canonical=joblib.load(complete['canonical_context_source'])
    roles=np.load(full/f'input/artifacts/fold{f}/indices.npz')
    assert len(roles['context'])==64000 and len(roles['correction'])==6000
    assert np.array_equal(np.sort(np.r_[roles['context'],roles['correction']]),np.arange(70000))
    assert not np.intersect1d(roles['context'],roles['correction']).size
    assert canonical['context'].application_id.tolist()==train.application_id.iloc[roles['context']].tolist()
    assert np.array_equal(canonical['context'].target.to_numpy(int),train.target.iloc[roles['context']].to_numpy(int))
    x=pd.read_parquet(root/'test_features.parquet')
    assert x.application_id.is_unique and len(x)==30000 and set(x.application_id)==set(sample.application_id)
    assert set(x.application_id).isdisjoint(train.application_id)
    sp=joblib.load(root/'spline.joblib')
    assert sp['fit_rows']==70000 and sp['fit_ids_sha']==b.ids_hash(train,np.arange(70000))
    expected_dims={'baseline':(18,77),'maturity':(18,77),'periodicity':(29,88)}[plan['variant']]
    assert len(sp['columns'])==expected_dims[1]
    spline=b.predict_spline(sp,x)
    parts=[]
    for bundle in complete['bundles']:
        head=joblib.load(root/bundle['head_file'])
        assert len(head['columns'])==expected_dims[0]
        assert head['spec']['correction_strength']==.5
        if bundle['raw_mode']=='original':raw=np.load(complete['original_raw_source'])
        else:raw=np.load(root/'raw_fresh.npz')
        assert raw['application_id'][6000:].tolist()==x.application_id.tolist()
        corrected=b.corrected_probability(head,x,raw['prediction'][6000:])[0]
        parts.append(.75*corrected+.25*spline)
    pred=np.mean(parts,axis=0)
    expected=np.load(root/'test_predictions.npz')
    error=float(np.max(np.abs(pred-expected['prediction'])))
    assert error<2e-14
    predictions.append(pd.Series(pred,index=x.application_id).reindex(sample.application_id).to_numpy())
    checks.append(dict(fold=f,models_replayed=True,context_labels_verified=True,roles_disjoint=True,full_training_coverage=True,head_models=len(parts),max_prediction_error=error))
mean=np.mean(predictions,axis=0)
csv=R/plan['submission_filename']
actual=pd.read_csv(csv)
assert actual.columns.tolist()==['application_id','default_probability']
assert actual.application_id.tolist()==sample.application_id.tolist()
assert np.isfinite(actual.default_probability).all()and actual.default_probability.between(0,1,inclusive='neither').all()
error=float(np.max(np.abs(actual.default_probability.to_numpy()-mean)))
assert error<1e-15
b.dump(dict(status='passed',checks=checks,CSV_rows=30000,sample_ID_order_verified=True,
            max_mean_prediction_error=error,submission_sha256=b.sha(csv),no_new_model_fit=True,
            verification_source_sha256=b.sha(Path(__file__))),R/'FINAL_INDEPENDENT_AUDIT.json')
