"""Rebuild missing per-fold V25 test scores from immutable saved checkpoints."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import corrected_probability,sha,dump
from experiments import combined

def main():
    result=json.loads((V/'runs/V25/RESULT.json').read_text());columns=result['columns']
    schema=pq.read_schema(ROOT/'data/test_features.parquet').names
    base=[c for c in columns if c in schema];extra=[c for c in columns if c not in schema]
    test=pd.read_parquet(ROOT/'data/test_features.parquet',columns=['application_id']+base)
    sample=pd.read_csv(ROOT/'data/sample_submission.csv');average=np.zeros(len(sample));audits=[]
    for fold in range(5):
        d=combined(test,pd.read_parquet(V/'extension/data'/f'fold{fold}/test.parquet',columns=['application_id']+extra))
        model=joblib.load(V/'runs/V25'/f'fold{fold}_correction.joblib');assert model['columns']==columns
        raw=pd.read_parquet(ROOT/'artifacts'/f'fold{fold}/kumo_test.parquet')
        spline=pd.read_parquet(ROOT/'artifacts'/f'fold{fold}/spline_test.parquet')
        assert d.application_id.tolist()==raw.application_id.tolist()==spline.application_id.tolist()
        corrected,delta=corrected_probability(model,d,raw.prediction.to_numpy())
        pred=.75*corrected+.25*spline.prediction.to_numpy()
        assert np.isfinite(pred).all() and np.all((pred>=0)&(pred<=1))
        output=V/'runs/V25'/f'fold{fold}_test.parquet'
        frame=pd.DataFrame(dict(application_id=d.application_id,prediction=pred,correction_logit=delta))
        if output.exists():
            saved=pd.read_parquet(output);assert saved.application_id.tolist()==frame.application_id.tolist();assert np.max(abs(saved.prediction.to_numpy()-pred))<1e-14
            pred=saved.prediction.to_numpy()
        else:frame.to_parquet(output,index=False)
        average+=pd.Series(pred,index=d.application_id).reindex(sample.application_id).to_numpy()/5
        audits.append(dict(fold=fold,rows=len(frame),correction_checkpoint_sha256=sha(V/'runs/V25'/f'fold{fold}_correction.joblib'),raw_Kumo_sha256=sha(ROOT/'artifacts'/f'fold{fold}/kumo_test.parquet'),cached_test_sha256=sha(output)))
    rebuilt=pd.DataFrame(dict(application_id=sample.application_id,default_probability=average)).to_csv(index=False).encode()
    import hashlib
    digest=hashlib.sha256(rebuilt).hexdigest();original=ROOT/result['submission'];assert sha(original)==result['submission_sha256']
    expected=pd.read_csv(original,float_precision='round_trip');assert expected.application_id.tolist()==sample.application_id.tolist()
    old=expected.default_probability.to_numpy();difference=float(np.max(abs(old-average)));order_same=bool(np.array_equal(np.argsort(old,kind='stable'),np.argsort(average,kind='stable')))
    assert difference<1e-14 and order_same,'Replayed baseline cache must preserve scores numerically and all test-score ordering'
    audit=dict(status='PASS',original_submission_unmodified=True,replay_submission_byte_identical=rebuilt==original.read_bytes(),original_submission_sha256=result['submission_sha256'],replayed_aggregate_sha256=digest,max_absolute_score_error=difference,identical_test_score_order=order_same,new_training=False,checkpoint_replay_only=True,public_score_assigned_to_replay=False,note='Four aggregate scores differed by<=5.55e-17 in the inspected replay; no claim of new bytes carrying the public score',folds=audits)
    dump(audit,V/'reports/V25_TEST_CACHE_REPLAY.json');print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
