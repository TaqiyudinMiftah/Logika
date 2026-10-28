"""Re-evaluate all four v3 components, with fold-local priors and shared CV rows."""
import json
import numpy as np
import pandas as pd
from .common import read,write,metrics,verify_oof
from .design import DEFAULT
from .pipeline import predict

def specs(root):
    v3=read(root/'vendor/v3_recipe.json');old_ext=read(root/'vendor/extended_manifest.json');old_cat=read(root/'vendor/compact_manifest.json')
    a=dict(v3['definitions']['price_lgb31_balanced_250'],blend_weight=1/3)
    b=dict(v3['definitions']['price_cat5_balanced_700'],blend_weight=1/3)
    c=dict(next(x for x in old_ext['candidates'] if x['name']=='lgb_leaves31'),columns=old_ext['columns'],blend_weight=1/12)
    d=dict(next(x for x in old_cat['candidates'] if x['name']=='cat_depth5'),columns=old_cat['columns'],blend_weight=1/4)
    return [a,b,c,d]

def train_predict(bank,spec,fitrows,y,queryrows,seed):
    cols=spec['columns'];numeric=[c for c in cols if c in bank.lookup];categorical=[c for c in cols if c in bank.manifest['categorical_columns']]
    def frame(rows):
        x,_=bank.cohort(fitrows,rows,numeric,DEFAULT);df=pd.DataFrame(x,columns=numeric)
        for c in categorical:df[c]=bank.categorical[c].iloc[rows].to_numpy(dtype=str)
        return df[cols]
    x=frame(fitrows);q=frame(queryrows);params=spec['params'];family=spec['family']
    if family=='cat':
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(iterations=spec['iterations'],learning_rate=.04,loss_function='Logloss',eval_metric='AUC',random_seed=seed,thread_count=1,verbose=False,allow_writing_files=False,**params)
        model.fit(x,y,cat_features=categorical);w=params.get('class_weights',[1,1]);w=w[1]/w[0]
    else:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from lightgbm import LGBMClassifier
        encoder=ColumnTransformer([('num','passthrough',numeric),('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
        x=encoder.fit_transform(x);q=encoder.transform(q)
        model=LGBMClassifier(n_estimators=spec['iterations'],learning_rate=.03,objective='binary',metric='None',subsample=.85,subsample_freq=1,colsample_bytree=.85,reg_alpha=.1,n_jobs=1,random_state=seed,verbosity=-1,**params)
        model.fit(x,y);w=params.get('scale_pos_weight',1)
    return predict(model,q,w)

def run(root,bank,task,folder):
    confirm=task.get('mode')=='reference_confirmation';all_specs=specs(root)
    if confirm:
        if 'reference_v3' not in read(root/'finalists.json')['ids']:raise ValueError('Reference not frozen')
        rows=np.load(bank.root/'confirmation_rows.npy');p=np.zeros(len(rows))
        for spec in all_specs:p+=spec['blend_weight']*train_predict(bank,spec,bank.dev,bank.y,rows,2026)
        np.save(folder/'prediction.npy',p);y=np.load(bank.root/'confirmation_y.npy')
        return {'kind':'confirmation','parent_id':'reference_v3','metrics':metrics(y,p)}
    p=np.full(len(bank.y),np.nan)
    for f in range(3):
        ti=np.flatnonzero(bank.folds!=f);vi=np.flatnonzero(bank.folds==f);pv=np.zeros(len(vi))
        for j,spec in enumerate(all_specs):
            pv+=spec['blend_weight']*train_predict(bank,spec,bank.dev[ti],bank.y[ti],bank.dev[vi],2026+f)
            print(json.dumps({'fold':f,'component':j,'rows':len(ti)}),flush=True)
        p[vi]=pv;np.save(folder/f'fold_{f}_prediction.npy',pv)
        print(json.dumps({'fold':f,'metrics':metrics(bank.y[vi],pv)}),flush=True)
    np.save(folder/'prediction.npy',p);np.savez_compressed(folder/'oof.npz',application_id=bank.metadata.application_id.iloc[bank.dev].to_numpy(dtype=str),target=bank.y,fold=bank.folds,prediction=p)
    write(folder/'recipe.json',all_specs)
    return {'kind':'reference','name':'locked_v3_refit','caveat':'All four components refit under campaign CV; historical v3 reused a prior 3-fold component. Cohort priors now fit on each training fold.',**verify_oof(bank.y,p,bank.folds)}
