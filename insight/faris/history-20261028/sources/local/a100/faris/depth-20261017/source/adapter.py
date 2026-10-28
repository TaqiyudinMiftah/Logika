"""Full three-fold depth search, with retained fold-local models and preprocessing."""
import json, os, time
from pathlib import Path
import numpy as np
from metaexp.features import Bank
from metaexp.pipeline import Preprocessor, predict
from metaexp.common import metrics, json_safe
import joblib

DEADLINE=1792191600.0  # 2026-10-17 06:00 Asia/Jakarta

def make_model(kind,p,seed,cpus,gpu):
    if kind=='linear':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=p['strength'],max_iter=1200,tol=1e-4,class_weight={0:1,1:p['weight']},random_state=seed)
    if kind=='lgb':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=p['trees'],num_leaves=p['leaves'],learning_rate=p['rate'],min_child_samples=p['minimum'],reg_lambda=p['strength'],colsample_bytree=p['columns'],scale_pos_weight=p['weight'],n_jobs=cpus,verbosity=-1,random_state=seed,max_bin=127)
    if kind=='cat':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=p['trees'],depth=p['depth'],learning_rate=p['rate'],l2_leaf_reg=p['strength'],class_weights=[1,p['weight']],thread_count=cpus,random_seed=seed,verbose=False,allow_writing_files=False,task_type='GPU' if gpu else 'CPU',**({'devices':'0','gpu_ram_part':.8} if gpu else {}))
    if kind=='xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=p['trees'],max_depth=p['depth'],learning_rate=p['rate'],min_child_weight=p['minimum'],reg_lambda=p['strength'],colsample_bytree=p['columns'],scale_pos_weight=p['weight'],n_jobs=cpus,random_state=seed,tree_method='hist',device='cuda:0' if gpu else 'cpu',max_bin=128)
    raise ValueError(kind)

def run_trial(ctx):
    if time.time()>=DEADLINE:raise RuntimeError('Absolute 06:00 WIB deadline')
    bank=Bank(Path(ctx.inputs['bank']).parent)
    split=json.loads(ctx.split.read_text())
    assert np.array_equal(bank.dev,np.asarray(split['development_rows']))
    assert np.array_equal(bank.folds,np.asarray(split['folds']))
    views=json.loads(Path(__file__).with_name('views.json').read_text())
    p=ctx.params;rep=json.loads(json.dumps(views[p['view']]))
    rep['config'].update(weight=p['weight'])
    kind=ctx.family.split('_')[0];cpus=ctx.spec['resources']['cpus'];gpu=ctx.spec['resources'].get('gpus',0)>0
    if p.get('variant','original')=='no_reduction':rep['config'].update(reducer='none')
    if p.get('variant')=='no_spline':rep['config'].update(spline=False)
    if p.get('variant')=='indicator':rep['config'].update(indicator=True)
    if p.get('variant')=='no_interactions':rep['config'].update(interactions=False)
    pred=np.full(len(bank.y),np.nan);fold_scores=[]
    for fold in range(3):
        ctx.check_stop();ti=np.flatnonzero(bank.folds!=fold);vi=np.flatnonzero(bank.folds==fold)
        pre=Preprocessor(bank,rep,'linear' if kind=='linear' else kind,ctx.seed+fold)
        x=pre.fit_transform(bank.dev[ti],bank.y[ti]);xv=pre.transform(bank.dev[vi])
        m=make_model(kind,p,ctx.seed+fold,cpus,gpu)
        m.fit(x,bank.y[ti]);pred[vi]=predict(m,xv,p['weight'])
        ctx.save_json(f'fold_{fold}/resolved.json',{'model':json_safe(m.get_params()),'preprocessing':pre.audit,'representation':rep,'seed':ctx.seed+fold,'host':os.uname().nodename,'gpu_visibility':os.environ.get('CUDA_VISIBLE_DEVICES')})
        pre.bank=None
        name=f'fold_{fold}/pipeline.joblib';joblib.dump({'preprocessing':pre,'model':m,'weight':p['weight'],'representation':rep},ctx.path(name),compress=1);ctx.model(name)
        np.savez_compressed(ctx.path(f'fold_{fold}/validation.npz'),rows=bank.dev[vi],prediction=pred[vi])
        fold_scores.append(metrics(bank.y[vi],pred[vi])['ks'])
        ctx.save_json('fold_progress.json',{'completed_folds':fold+1,'per_fold':fold_scores})
        del x,xv,pre,m
    np.savez_compressed(ctx.path('validation.npz'),rows=bank.dev,fold=bank.folds,prediction=pred)
    return {'validation':['validation.npz'],'folds':3,'selection_limit':'Development folds reused for adaptive historical selection; no independent confirmation claimed.'}

def evaluate(ctx,result):
    bank=Bank(Path(ctx.inputs['bank']).parent);z=np.load(ctx.path('validation.npz'))
    assert np.array_equal(z['rows'],bank.dev) and np.array_equal(z['fold'],bank.folds)
    p=z['prediction'];assert p.shape==bank.y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    from scipy.stats import ks_2samp
    score=metrics(bank.y,p);ind=float(ks_2samp(p[bank.y==0],p[bank.y==1]).statistic)
    assert abs(score['ks']-ind)<1e-12
    per=[metrics(bank.y[bank.folds==f],p[bank.folds==f])['ks'] for f in range(3)]
    return {'score':score['ks'],'per_fold':per,'metrics':score,'independent_ks':ind,'protocol':'development-only, historical/adaptive reuse; not external or untouched confirmation'}
