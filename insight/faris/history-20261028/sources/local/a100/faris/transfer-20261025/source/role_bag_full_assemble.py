"""Assemble selected H010 full refit from saved member predictions."""
import time
import numpy as np
import pandas as pd
import baseline as b
from robustness import R,A,read
D=R/'candidate_F';plan=read(D/'FULL_REFIT_PLAN.json')
assert time.time()<plan['submission_deadline_epoch']
sample=pd.read_csv(A/'exploration-12h-20261017/raw/sample_submission(3).csv')
assert len(sample)==30000 and sample.application_id.is_unique
parts=[]
for f in range(5):
    root=D/f'final/fold{f}';complete=read(root/'complete.json');assert complete['status']=='complete'
    for n,h in complete['files'].items():assert b.sha(root/n)==h
    a=np.load(root/'test_predictions.npz');assert len(set(a['application_id']))==30000
    assert set(a['application_id'])==set(sample.application_id)
    parts.append(pd.Series(a['prediction'],index=a['application_id']).reindex(sample.application_id).to_numpy())
pred=np.mean(parts,axis=0);assert np.isfinite(pred).all()and((pred>0)&(pred<1)).all()
path=D/plan['submission_filename'];assert not path.exists()
pd.DataFrame(dict(application_id=sample.application_id,default_probability=pred)).to_csv(path,index=False)
actual=pd.read_csv(path);assert actual.columns.tolist()==['application_id','default_probability']
assert actual.application_id.tolist()==sample.application_id.tolist()
error=float(np.max(np.abs(actual.default_probability.to_numpy()-pred)));assert error<1e-15
b.dump(dict(status='passed',submission_sha256=b.sha(path),rows=30000,runs=5,context_rows_per_run=64000,correction_rows_per_run=6000,shared_prior_rows=70000,spline_rows=70000,every_training_row_used=True,CSV_readback_max_error=error,final_refit_validation_score=None,research_recipe_OOF_KS=plan['research_pooled_OOF_KS'],uploaded=False,source_sha256=b.sha(__file__)),D/'FULL_REFIT_DELIVERY.json')
