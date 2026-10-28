"""One bounded expert fit; no correction labels or outer labels train the expert."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import socket
import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common import R,load,dump,sha,ids_sha,utc,check_seal
sys.path.insert(0,str(R/'source/vendor'))


def cpu_fit(family,frame,ctx,y,query,seed):
    train,stop=train_test_split(np.arange(len(ctx)),test_size=.15,stratify=y,random_state=seed)
    x=frame.copy();categorical=x.select_dtypes(include=['object','string','category']).columns.tolist()
    if family=='catboost':
        from catboost import CatBoostClassifier
        for c in categorical:x[c]=x[c].astype('string').fillna('MISSING').astype(str)
        params=dict(iterations=1600,depth=6,learning_rate=.035,l2_leaf_reg=10,loss_function='Logloss',
                    random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
        pilot=CatBoostClassifier(**params)
        pilot.fit(x.iloc[ctx[train]],y[train],cat_features=categorical,
                  eval_set=(x.iloc[ctx[stop]],y[stop]),early_stopping_rounds=100)
        iterations=max(1,pilot.best_iteration_+1);del pilot;gc.collect()
        params['iterations']=iterations;model=CatBoostClassifier(**params)
        model.fit(x.iloc[ctx],y,cat_features=categorical)
    else:
        import lightgbm as lgb
        for c in categorical:
            vocabulary=sorted(x[c].iloc[ctx].astype('string').fillna('MISSING').unique().tolist())
            x[c]=pd.Categorical(x[c].astype('string').fillna('MISSING'),categories=vocabulary)
        params=dict(n_estimators=2200,num_leaves=15,learning_rate=.025,max_depth=6,
                    min_child_samples=100,reg_lambda=15,subsample=.9,colsample_bytree=.9,
                    random_state=seed,n_jobs=2,verbosity=-1)
        pilot=lgb.LGBMClassifier(**params)
        pilot.fit(x.iloc[ctx[train]],y[train],eval_set=[(x.iloc[ctx[stop]],y[stop])],
                  callbacks=[lgb.early_stopping(120,verbose=False)])
        iterations=max(1,pilot.best_iteration_);del pilot;gc.collect()
        params['n_estimators']=iterations;model=lgb.LGBMClassifier(**params)
        model.fit(x.iloc[ctx],y)
    a=model.predict_proba(x.iloc[query])[:,1]
    b=model.predict_proba(x.iloc[query[:256]])[:,1]
    assert np.max(abs(a[:len(b)]-b))<1e-12
    return a,dict(parameters=params,inner_train_rows=len(train),stopping_rows=len(stop),
                  refit_context_rows=len(ctx),stopping_rows_sha256=ids_sha(ctx[stop]),
                  correction_used_for_early_stopping=False)


def main(task_id,attempt,pilot=False):
    started=time.time();clock=check_seal();config=load('CONFIG.json');task=config['tasks'][task_id]
    if not pilot:
        gate=load('GATE.json');assert gate['status']=='PASS'
        assert gate['contract_sha256']==clock['contract_sha256'] and gate['source_seal_sha256']==sha(R/'SOURCE_SEAL.json')
    else:
        assert task_id in config['pilot_tasks']
    assert time.time()<clock['admission_end_epoch']
    folder=R/f'runs/{task_id}/attempt{attempt}';folder.mkdir(parents=True,exist_ok=False)
    try:
        fold=task['fold'];packet=R/f'packets/fold{fold}';m=json.loads((packet/'MANIFEST.json').read_text())
        expected=load('PREPARATION_AUDIT.json')if pilot else gate
        assert sha(packet/'MANIFEST.json')==expected['packet_manifests'][str(fold)]
        assert sha(packet/'features.parquet')==m['feature_sha256'] and sha(packet/'roles.npz')==m['roles_sha256']
        roles=np.load(packet/'roles.npz');ctx=roles['context'];query=np.r_[roles['correction'],roles['validation']]
        columns=m['views'][task['view']];frame=pd.read_parquet(packet/'features.parquet',columns=columns)
        limit=400 if task['view']=='expanded_union' and task['family']!='kumo' else 99
        assert 0<len(columns)<=limit and not {'target','application_id','customer_id'}&set(columns)
        assert not np.intersect1d(ctx,query).size
        seed=task['seed']+fold
        record=dict(task=task,attempt=attempt,pilot=pilot,seed=seed,columns=columns,
                    packet_manifest_sha256=sha(packet/'MANIFEST.json'),context_rows_sha256=ids_sha(ctx),
                    prediction_rows_sha256=ids_sha(query),context_rows=len(ctx),query_rows=len(query),
                    context_labels_sha256=ids_sha(roles['y_context']),source_sha256=sha(__file__),
                    source_seal_sha256=sha(R/'SOURCE_SEAL.json'),contract_sha256=clock['contract_sha256'],
                    correction_labels_used=False,outer_labels_loaded=False,
                    UTC=utc(),host=socket.gethostname(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES',''),
                    versions={k:importlib.metadata.version(k)for k in ['numpy','pandas','pyarrow','scipy','scikit-learn','lightgbm','catboost','torch','structured-data-models']})
        dump(record,folder/'START.json')
        if task['family']=='kumo':
            from kumo_engine import predict
            paths=load('PATHS.json')
            p,audit=predict(frame.iloc[ctx],frame.iloc[query],columns,roles['y_context'],paths['checkpoint'],
                      seed,engine='concat_context_vocab',estimators=8,pred_batch=256,mem_fraction=.85)
            audit.update(size='large',estimators=8,encoding='context vocabulary',precision='CUDA AMP float16',
                         checkpoint_sha256=sha(paths['checkpoint']),execution_backend='cuda',correction_labels_used=False)
        else:
            p,audit=cpu_fit(task['family'],frame,ctx,roles['y_context'],query,seed)
        assert len(p)==20000 and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        np.savez_compressed(folder/'prediction.npz',rows=query,raw=np.asarray(p,float))
        dump(dict(status='complete',task_id=task_id,seconds=time.time()-started,UTC=utc(),audit=audit,
                  prediction_sha256=sha(folder/'prediction.npz'),start_sha256=sha(folder/'START.json'),
                  max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  weights_saved=False,KV_cache_saved=False),folder/'RESULT.json')
        print(json.dumps(dict(task_id=task_id,status='complete',seconds=time.time()-started)),flush=True)
    except BaseException:
        dump(dict(status='failed',task_id=task_id,attempt=attempt,UTC=utc(),error=traceback.format_exc(),
                  seconds=time.time()-started,scientific_conclusion='none; execution failure'),folder/'FAILURE.json')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--pilot',action='store_true');a=parser.parse_args();main(a.task,a.attempt,a.pilot)
