"""Reuse exact V80 values; no event rereading or nuisance refitting."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'


def main():
    parent=json.loads((V/'runs/V80/RESULT.json').read_text());names=parent['new_features']
    raw=V/'behavior/data/raw_features.parquet'
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    assert digest==parent['behavior_feature_matrix_sha256']
    data=pd.read_parquet(raw,columns=['application_id']+names)
    assert len(data)==100000 and data.application_id.is_unique
    lookup=data.set_index('application_id');checks=[]
    for fold in range(5):
        val=pd.read_parquet(V/'behavior/data'/f'fold{fold}'/'validation.parquet',columns=['application_id']+names)
        expected=lookup.reindex(val.application_id)[names].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected,val[names].reset_index(drop=True),check_exact=True,check_dtype=False)
        checks.append(dict(fold=fold,exact_value_parity=True,rows=len(val)))
    out=V/'light_fe/data/spectral_features.parquet';out.parent.mkdir(exist_ok=True,parents=True);data.to_parquet(out,index=False)
    report=dict(status='COMPLETE',source='V80 saved behavioral spectral features',raw_source_sha256=digest,columns=names,rows=len(data),fold_checks=checks,no_default_labels_used=True,raw_event_features_changed=False,feature_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (V/'light_fe/reports/SPECTRAL_CACHE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
