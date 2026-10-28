"""Reuse V25 inputs; verify deterministic counts across all cached fold roles."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'


def main():
    parent=json.loads((V/'runs/V25/RESULT.json').read_text());columns=parent['columns']
    schema=pq.read_schema(ROOT/'data/train_features.parquet').names
    basecols=[c for c in columns if c in schema];extra=[c for c in columns if c not in schema]
    train=pd.read_parquet(ROOT/'data/train_features.parquet',columns=['application_id']+basecols)
    test=pd.read_parquet(ROOT/'data/test_features.parquet',columns=['application_id']+basecols)
    base=pd.concat([train[['application_id']+basecols],test[['application_id']+basecols]],ignore_index=True).set_index('application_id')
    pieces=[pd.read_parquet(V/'extension/data'/f'fold{i}'/'validation.parquet',columns=['application_id']+extra) for i in range(5)]
    pieces.append(pd.read_parquet(V/'extension/data/fold0/test.parquet',columns=['application_id']+extra))
    counts=pd.concat(pieces,ignore_index=True).set_index('application_id');assert counts.index.is_unique and set(counts.index)==set(base.index)
    checks=[]
    for i in range(5):
        for role in ['correction','validation','test']:
            d=pd.read_parquet(V/'extension/data'/f'fold{i}'/(role+'.parquet'),columns=['application_id']+extra).set_index('application_id')
            assert np.array_equal(d[extra].to_numpy(),counts.loc[d.index,extra].to_numpy(),equal_nan=True)
            checks.append(dict(fold=i,role=role,rows=len(d)))
    assert np.isfinite(counts.to_numpy(float)).all() and np.all(counts.to_numpy(float)>=0)
    base=base.join(counts,validate='one_to_one')[columns].reset_index();assert len(base)==100000
    out=V/'anchored_spline/data/inputs.parquet';out.parent.mkdir(parents=True,exist_ok=True);base.to_parquet(out,index=False)
    report=dict(status='COMPLETE',rows=len(base),features=columns,new_historical_features=False,
        target_read_for_feature_construction=False,output_contains_target=False,
        cross_fold_count_invariance=True,count_columns=extra,checks=checks,
        inherited_baseline_feature_provenance=True,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (V/'anchored_spline/reports').mkdir(parents=True,exist_ok=True)
    (V/'anchored_spline/reports/FEATURE_BUILD.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=len(base),features=len(columns),invariance=True)),flush=True)


if __name__=='__main__':main()
