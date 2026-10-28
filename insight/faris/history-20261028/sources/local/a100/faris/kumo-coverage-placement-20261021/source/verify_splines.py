"""Replay every refitted shared spline from retained preprocessing and weights."""
from pathlib import Path
import json
import numpy as np
import joblib
import experiment as ex
ROOT=Path(__file__).resolve().parents[1]
def main():
    checks=[]
    for f in range(5):
        _,inp,data,_=ex.load(f);ix=ex.indices(inp,data,f);root=ROOT/f'shared/fold{f}'
        complete=ex.read(root/'complete.json')
        for name,digest in complete['files'].items():assert ex.b.sha(root/name)==digest
        model=joblib.load(root/'spline.joblib');assert model['fit_ids_sha']==ex.b.ids_hash(data,ix['train'])
        assert model['fit_rows']==56000
        saved=np.load(root/'validation.npz');assert np.array_equal(saved['rows'],ix['validation'])
        query=data.iloc[ix['validation']].drop(columns='target')
        actual=ex.b.predict_spline(model,query);error=float(np.max(abs(actual-saved['prediction'])))
        assert error<1e-12,error
        checks.append(dict(fold=f,training_rows=56000,validation_rows=14000,
            validation_labels_in_predictor_input=False,max_prediction_error=error,
            model_sha256=ex.b.sha(root/'spline.joblib')))
    ex.write(ROOT/'SPLINE_AUDIT.json',dict(status='passed',retained_models_replayed=5,
        full_validation_rows=70000,checks=checks))
    print(json.dumps(dict(status='passed',models=5,max_error=max(c['max_prediction_error']for c in checks))))
if __name__=='__main__':main()
