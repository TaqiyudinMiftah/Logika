"""Bounded fold0 replay of the exact spline77/88, H016 and spline200 implementations."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from assembly_runner import sha,dump,spline200_columns
from spline_models import fixed_spline_fit,direct_spline_update,make_scaled_spline_matrix,fit_logistic_irls,predict_irls


def main():
    home=Path('/mnt/home/faris');exp=home/'logika-faris-exploration';frz=home/'logika-frozen-shadow-20261026';joint=home/'logika-joint-20261028'
    root=home/'logika-antigravity-refit-20261028/assembly/spline_replay/fold0';root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(exp/'transfer-20261025/source'))
    import baseline,feature_head
    started=time.time();fold=0
    source=[Path(__file__),Path(__file__).with_name('spline_models.py'),Path(__file__).with_name('assembly_runner.py')]
    dump(root/'PLAN.json',{'status':'planned','fold':fold,'fit_size':56000,'source_sha256':{str(p):sha(p)for p in source},
         'fixed_spline_prediction_max_abs_error_acceptance':1e-10,'H016_prediction_max_abs_error_acceptance':1e-10,
         'spline200_cached_float32_max_abs_error_acceptance':1e-7,'weights_saved':False,'scientific_selection_allowed':False,
         'historical_bundle_policy':'Reconstruct columns from retained metadata; no historical fitted bundles required.'})
    original=pd.read_parquet(exp/'ranking-pipeline-20261020/input/data/train_features.parquet',columns=['application_id','target'])
    ix=np.load(exp/'ranking-pipeline-20261020/input/artifacts/fold0/indices.npz')
    metadata=json.loads((exp/'ranking-pipeline-20261020/input/artifacts/fold0/spline_metadata.json').read_text())
    cols=metadata['recipe']['spec']['columns']
    table=pd.read_parquet(exp/'fold-local-priors-20261021/fold0/features.parquet',columns=['application_id']+cols).set_index('application_id').loc[original.application_id].reset_index()
    mature=pd.read_parquet(exp/'repayment-severity-fe-20261021/fold0/features.parquet').set_index('application_id').loc[original.application_id]
    for t in [1,30]:table[f're_late_{t}_posterior']=mature[f'matched_beta_late_{t}'].to_numpy(float)
    from assembly_runner import SPECTRAL
    spectral=pd.read_parquet(exp/'shopping-periodicity-fe-20261021/true_features.parquet',columns=['application_id']+SPECTRAL).set_index('application_id').loc[original.application_id]
    for c in SPECTRAL:table[c]=spectral[c].to_numpy(float)
    training=table.iloc[ix['train']].copy();training['target']=original.target.iloc[ix['train']].to_numpy(int)
    query=table.iloc[ix['validation']].copy();results={};predictions={}
    baseline_npz=np.load(frz/'frozen/fold0/baseline.npz')
    # Reconstruct the true saved periodicity spline from the exact original
    # correction query used by its source, rather than AGY's float32-expanded
    # head surrogate. The latter intentionally remains in ALL66 route replay.
    exact_query=pd.read_parquet(exp/'kumo-temperature-adaptation-20261021/full_data/fold0/query.parquet')
    mature_ids=mature.copy();mature_ids.index=original.application_id
    spectral_ids=spectral.copy();spectral_ids.index=original.application_id
    for t in [1,30]:exact_query[f're_late_{t}_posterior']=mature_ids.loc[exact_query.application_id,f'matched_beta_late_{t}'].to_numpy(float)
    for c in SPECTRAL:exact_query[c]=spectral_ids.loc[exact_query.application_id,c].to_numpy(float)
    head18=json.loads((frz/'frozen/fold0/MANIFEST.json').read_text())['head_columns']
    h29=feature_head.fit(exact_query.iloc[:6000].assign(target=original.target.iloc[ix['correction']].to_numpy(int)),baseline_npz['raw'][:6000],head18+SPECTRAL,.003,.5)
    corrected=baseline.corrected_probability(h29,exact_query.iloc[6000:],baseline_npz['raw'][6000:])[0]
    with np.load(exp/'transfer-20261025/ROBUSTNESS_OOF.npz')as z:refper=(z['periodicity__original'][ix['validation']]-.75*corrected)/.25
    for name,columns in [('mat77',cols),('per88',cols+SPECTRAL)]:
        fitted=fixed_spline_fit(training,columns,baseline.prepare)
        pred=baseline.predict_spline(fitted,query);ref=baseline_npz['spline']if name=='mat77'else refper
        error=float(np.max(np.abs(pred-ref)));assert error<1e-10,(name,error)
        results[name]={'max_absolute_error':error,'parameters':fitted['resolved_parameters'],'fit_rows':len(training)}
        predictions[name]=pred
        if name=='per88':periodic_model=fitted
    direct,optimizer=direct_spline_update(periodic_model,training,query,baseline.prepare)
    with np.load(exp/'transfer-20261025/spline_direct_ks/fold0/validation.npz')as z:ref=refper+z['spline_delta']
    error=float(np.max(np.abs(direct-ref)));assert error<1e-10,error
    results['H016']={'max_absolute_error':error,'optimizer':optimizer};predictions['H016']=direct
    expanded=pd.read_parquet(joint/'fold0/expanded.parquet')
    with np.load(frz/'frozen/fold0/roles.npz')as z:rows=np.r_[z['context'],z['correction']];validation=z['validation']
    feature_cols=spline200_columns(frz,joint)
    with threadpool_limits(limits=4):
        X=make_scaled_spline_matrix(expanded,rows,feature_cols,n_knots=6,degree=2,top_inter_n=35)
        intercept,weights=fit_logistic_irls(X[rows],original.target.iloc[rows].to_numpy(int),C=.0025)
        pred,_=predict_irls(X[validation],intercept,weights)
    with np.load(joint/'linear56k_runs.npz')as z:ref=z['val_spline200'][validation]
    error=float(np.max(np.abs(pred-ref)));assert error<1e-7,error
    results['spline200']={'max_absolute_error':error,'comparison_storage':'cachedfloat32','feature_inputs':len(feature_cols),'design_columns':X.shape[1]};predictions['spline200']=pred
    np.savez_compressed(root/'prediction.npz',rows=validation,**predictions)
    dump(root/'complete.json',{'status':'passed','fold':fold,'seconds':time.time()-started,'checks':results,'PLAN_sha256':sha(root/'PLAN.json'),'prediction_sha256':sha(root/'prediction.npz'),'weights_saved':False,'independent_confirmation':False})
    print(json.dumps({'status':'passed','seconds':time.time()-started,'checks':{name:value['max_absolute_error']for name,value in results.items()}}),flush=True)


if __name__=='__main__':main()
