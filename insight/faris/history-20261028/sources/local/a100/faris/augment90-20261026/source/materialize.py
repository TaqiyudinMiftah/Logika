"""Keep every original baseline column; append at most ten frozen additions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
def digest(v):
    return hashlib.sha256(pd.util.hash_pandas_object(pd.Series(v),index=False).values.tobytes()).hexdigest()
def materialize(f,group):
    selection=json.loads((R/f'selection/fold{f.fold}.json').read_text())
    baseline=selection['baseline_features']
    assert len(baseline)==90
    if group=='replay89':
        ids=json.loads((R/'replay_reference/START.json').read_text())['columns']
        assert len(ids)==89 and set(ids)<set(baseline)
    else:
        additions=[] if group=='baseline90' else selection['groups'][group]
        assert len(additions)<=10 and not set(additions)&set(baseline)
        ids=baseline+additions
    frame=pd.DataFrame({fid:f.compute(f.specs[fid]) for fid in ids})
    assert len(frame)==70000 and len(frame.columns)<=100
    if group!='replay89':
        assert list(frame)[:90]==baseline
        seen={digest(frame[fid].to_numpy()[f.ctx]) for fid in baseline}
        for fid in ids[90:]:
            values=frame[fid].to_numpy()[f.ctx]
            assert pd.Series(values).nunique(dropna=False)>1
            h=digest(values);assert h not in seen;seen.add(h)
    return frame,{}
