"""Tree expert on the union of input information already present in V25.

Expert training uses56k outer-training applications, without Kumo prediction
inputs. Early stopping uses a fresh inner split, never outer validation labels.
"""
import gc
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import dump,sha,scores,ids_hash


def matrix(frame,columns,reference=None):
    cats=frame[columns].select_dtypes(include=['object','string','category']).columns.tolist()
    if reference is None:reference={c:sorted(frame[c].fillna('MISSING').astype(str).unique()) for c in cats}
    values=[];cat_indices=[]
    for j,c in enumerate(columns):
        if c in reference:
            values.append(frame[c].fillna('MISSING').astype(str).map({x:i for i,x in enumerate(reference[c])}).to_numpy(float))
            cat_indices.append(j)
        else:values.append(pd.to_numeric(frame[c],errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(float))
    return np.column_stack(values).astype(np.float32),reference,cat_indices


def ensure(key,loaded):
    tr_small,te_small,sample,y,folds,basepred,frames,parent=loaded
    out=V/'modeling/experts'/key;out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((ROOT/'BASELINE_LOCK.json').read_text())
    columns=list(dict.fromkeys(lock['Kumo']['features']+lock['original_spline']['features']+parent['columns']))
    tr=pd.read_parquet(ROOT/'data/train_features.parquet');te=pd.read_parquet(ROOT/'data/test_features.parquet')
    support=parent['support_features'];additional=parent['new_features']
    assert additional==['risk_conditional_negative_months','risk_conditional_nonnegative_months']
    common=pd.read_parquet(V/'data/features_raw.parquet',columns=['application_id']+support)
    count=pd.read_parquet(V/'extension/data/raw_features.parquet',columns=['application_id','ext_aux_allocation_negative_n','ext_aux_allocation_nonnegative_n']).rename(columns={'ext_aux_allocation_negative_n':additional[0],'ext_aux_allocation_nonnegative_n':additional[1]})
    common=common.merge(count,on='application_id',validate='one_to_one')
    tr=tr.merge(common,on='application_id',validate='one_to_one');te=te.merge(common,on='application_id',validate='one_to_one')
    assert tr.application_id.tolist()==tr_small.application_id.tolist() and te.application_id.tolist()==te_small.application_id.tolist()
    assert len(columns)==116 and all(c in tr for c in columns)
    config=dict(learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=150,l2_regularization=20.,max_iter=400)
    if key=='hgb_b':config.update(learning_rate=.04,max_leaf_nodes=15,min_samples_leaf=150,l2_regularization=30.,max_iter=500)
    assert key in ['hgb_a','hgb_b']
    predictions={};audits=[]
    for f in frames:
        fold=f['fold'];a=ROOT/'artifacts'/f'fold{fold}';idx=np.load(a/'indices.npz');ti,vi=idx['train'],idx['validation']
        assert not np.intersect1d(ti,vi).size and len(ti)==56000
        for c in support+additional:
            np.testing.assert_equal(tr.iloc[vi][c].to_numpy(),f['val'][c].to_numpy())
        cache=out/f'fold{fold}.joblib';predfile=out/f'fold{fold}_predictions.npz';receipt=out/f'fold{fold}_audit.json'
        if cache.exists() and predfile.exists() and receipt.exists():
            saved=np.load(predfile);predictions[fold]={k:saved[k] for k in saved};audits.append(json.loads(receipt.read_text()));continue
        inner_train,inner_valid=train_test_split(ti,test_size=.15,stratify=y[ti],random_state=44327+fold)
        xh,ref,cat_idx=matrix(tr.iloc[inner_train],columns)
        xv,_,_=matrix(tr.iloc[inner_valid],columns,ref)
        # Explicit arrays keep preprocessing and early stopping inside the inner split.
        tune=HistGradientBoostingClassifier(**config,categorical_features=cat_idx,early_stopping=True,validation_fraction=None,n_iter_no_change=25,tol=1e-6,random_state=2026+fold)
        with threadpool_limits(limits=2):tune.fit(xh,y[inner_train],X_val=xv,y_val=y[inner_valid])
        iterations=int(tune.n_iter_)
        del tune,xh,xv
        xtrain,ref,cat_idx=matrix(tr.iloc[ti],columns);xval,_,_=matrix(tr.iloc[vi],columns,ref);xtest,_,_=matrix(te,columns,ref)
        model=HistGradientBoostingClassifier(**dict(config,max_iter=iterations),categorical_features=cat_idx,early_stopping=False,random_state=2026+fold)
        with threadpool_limits(limits=2):
            model.fit(xtrain,y[ti]);pv=model.predict_proba(xval)[:,1];pt=model.predict_proba(xtest)[:,1];ph=model.predict_proba(xtrain)[:,1]
        audit=dict(fold=fold,training_rows=56000,validation_rows=14000,inner_train_rows=len(inner_train),inner_validation_rows=len(inner_valid),selected_iterations=iterations,train_ids_sha256=ids_hash(tr,ti),validation_ids_sha256=ids_hash(tr,vi),training_metrics=scores(y[ti],ph),validation_metrics=scores(y[vi],pv),outer_validation_labels_used_in_training=False,existing_feature_inputs=columns,new_historical_features=False)
        joblib.dump(dict(model=model,category_reference=ref,columns=columns,configuration=config,selected_iterations=iterations),cache,compress=3)
        np.savez_compressed(predfile,validation=pv,test=pt)
        dump(audit,receipt);predictions[fold]=dict(validation=pv,test=pt);audits.append(audit)
        print('EXPERT_FOLD_READY',key,fold,'iterations',iterations,'KS',audit['validation_metrics']['KS'],flush=True)
        del model,xtrain,xval,xtest,ph;gc.collect()
    dump(dict(status='READY',key=key,columns=columns,configuration=config,folds=audits,source_sha256=sha(__file__),point_in_time_certified=False,inherited_snapshot_uncertainty=True),out/'AUDIT.json')
    return predictions,audits
