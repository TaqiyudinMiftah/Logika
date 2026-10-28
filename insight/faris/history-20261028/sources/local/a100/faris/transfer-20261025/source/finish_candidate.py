"""Assemble a frozen candidate from five completed saved runs, with CSV readback."""
import argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
import baseline as b
from robustness import R,A,read

p=argparse.ArgumentParser();p.add_argument('--directory',default='.');args=p.parse_args()
root=R/args.directory
plan=read(root/'FULL_REFIT_PLAN.json')
assert time.time()<plan['submission_deadline_epoch']
sample=pd.read_csv(A/'exploration-12h-20261017/raw/sample_submission(3).csv')
assert len(sample)==30000 and sample.application_id.is_unique
predictions=[]
for f in range(5):
    run=root/f'final/fold{f}';complete=read(run/'complete.json')
    assert complete['status']=='complete'
    for n,h in complete['files'].items():assert b.sha(run/n)==h,n
    a=np.load(run/'test_predictions.npz')
    assert len(a['application_id'])==len(set(a['application_id']))==30000
    assert set(a['application_id'])==set(sample.application_id)
    predictions.append(pd.Series(a['prediction'],index=a['application_id']).reindex(sample.application_id).to_numpy())
prediction=np.mean(predictions,axis=0)
assert np.isfinite(prediction).all()and((prediction>0)&(prediction<1)).all()
output=root/plan['submission_filename'];assert not output.exists()
pd.DataFrame(dict(application_id=sample.application_id,default_probability=prediction)).to_csv(output,index=False)
actual=pd.read_csv(output)
assert actual.columns.tolist()==['application_id','default_probability']
assert actual.application_id.tolist()==sample.application_id.tolist()
error=float(np.max(np.abs(actual.default_probability.to_numpy()-prediction)));assert error<1e-15
b.dump(dict(status='passed',submission=str(output),submission_sha256=b.sha(output),rows=30000,
            runs=5,context_rows_per_run=64000,correction_rows_per_run=6000,spline_rows_per_run=70000,
            every_training_row_used=True,CSV_readback_max_error=error,
            final_refit_validation_score=None,research_recipe_OOF_KS=plan['research_pooled_OOF_KS'],
            uploaded=False,source_sha256=b.sha(Path(__file__))),root/'FULL_REFIT_DELIVERY.json')
