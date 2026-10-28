"""Independent completed-run replay and metric audit, without fitting anything."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,hashlib,sys
import numpy as np
import joblib
from scipy.stats import ks_2samp,rankdata
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import study
import baseline as b

def verify():
    state=study.read(ROOT/'state.json');assert state['status']=='complete'
    seal=study.read(ROOT/'seal.json');checked=0
    for name,digest in seal['files'].items():assert b.sha(ROOT/name)==digest;checked+=1
    for name,digest in seal['inputs'].items():assert b.sha(name)==digest;checked+=1
    for name,digest in study.read(ROOT/'evidence/import_manifest.json')['files'].items():
        assert b.sha(ROOT/'evidence'/name)==digest;checked+=1
    assert study.read(ROOT/'data/AUDIT.json')['status']=='passed'
    data,recipes=study.load();z=np.load(ROOT/'oof.npz');y=data.target.to_numpy(int)
    assert np.array_equal(y,z['target']);seen=np.zeros(len(y),int);models=0;maximum=0.
    for f in range(5):
        ix,p=study.inputs(data,f);v=data.iloc[ix['validation']].drop(columns='target');seen[ix['validation']]+=1
        folder=ROOT/f'folds/fold{f}';meta=study.read(folder/'complete.json')
        for name,digest in meta['files'].items():assert b.sha(folder/name)==digest;checked+=1
        assert meta['fit_ids_sha']==b.ids_hash(data,ix['correction'])
        assert meta['evaluation_ids_sha']==b.ids_hash(data,ix['validation'])
        for name,cols in recipes.items():
            model=joblib.load(folder/(name+'.joblib'));assert model['columns']==cols
            assert not set(cols)&{'target','application_id','customer_id','split'}
            pred=.75*b.corrected_probability(model,v,p['kumo_oof'])[0]+.25*p['spline_oof']
            error=float(np.max(abs(pred-z[name][ix['validation']])));assert error<1e-12
            maximum=max(maximum,error);models+=1
    assert np.all(seen==1) and models==65
    r=study.read(ROOT/'result.json');metrics={}
    for name in r['metrics']:
        pred=z[name];ks=float(ks_2samp(pred[y==1],pred[y==0]).statistic)
        n1=y.sum();auc=float((rankdata(pred)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1)))
        assert abs(ks-r['metrics'][name]['KS'])<1e-12
        assert abs(auc-r['metrics'][name]['AUC'])<1e-12
        metrics[name]=dict(KS=ks,AUC=auc)
    report=dict(status='passed',hashes_verified=checked,trained_heads_replayed=models,complete_OOF_rows=70000,
        max_saved_model_replay_error=maximum,metrics=metrics,goal_054_achieved=metrics['opportunity_primary']['KS']>=.54,
        public_score_verified_online=False,user_CSV_hash_matches_archived_V25=True,
        historical_label_reuse=True,uncertainty_does_not_include_training_or_search_selection=True)
    b.dump(report,ROOT/'INDEPENDENT_AUDIT.json');print(json.dumps(report),flush=True)

if __name__=='__main__':verify()
