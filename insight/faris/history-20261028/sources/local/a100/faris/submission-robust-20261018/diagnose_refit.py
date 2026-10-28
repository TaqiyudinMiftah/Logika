"""Compare saved development-trained and final models on identical test covariates.

No fitting, no labels from competition test, and no submission selection here.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
FINAL=ROOT.parent/'submission-final-20261018'
sys.path.insert(0,str(FINAL))
from pipeline import read,write,sha,get_bank,predict_base


def main():
    plan=read(FINAL/'plan.json')
    os.environ['CUDA_VISIBLE_DEVICES']=plan['gpu_uuid']
    bank=get_bank(plan);rows=np.flatnonzero(bank.metadata.split.to_numpy()=='test')
    arrays={};base_cache={}
    import joblib
    for key,ref in plan['references'].items():
        folder=Path(ref['folder'])
        if key in plan['stream_configs']:
            arrays[key]=predict_base(plan['stream_configs'][key],folder/'confirmation',rows,plan)
        else:
            for cid in plan['shared_stack_components']:
                if cid not in base_cache:
                    base_cache[cid]=predict_base(cid,folder/'outer_0'/('component_'+cid)/'refit',rows,plan)
            model=joblib.load(folder/'outer_0/meta.joblib')
            arrays[key]=np.clip(model.predict(np.column_stack([base_cache[cid] for cid in plan['shared_stack_components']])),1e-8,1-1e-8)
    full=np.load(FINAL/'artifacts/streams.npz');assert np.array_equal(full['rows'],rows)
    result={}
    for key,p in arrays.items():
        q=full[key]
        result[key]={'rank_correlation':float(spearmanr(p,q).statistic),
            'mean_absolute_change':float(np.mean(np.abs(p-q))),
            'p99_absolute_change':float(np.quantile(np.abs(p-q),.99)),
            'development_fit_mean':float(p.mean()),'full_refit_mean':float(q.mean())}
    np.savez_compressed(ROOT/'evidence/development_models_test_predictions.npz',rows=rows,**arrays)
    write(ROOT/'evidence/refit_stability.json',{'same_test_rows':len(rows),'no_test_labels':True,
        'components':result,'interpretation':'Predictive instability diagnostic only; cannot determine which predictions are more accurate.'})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
