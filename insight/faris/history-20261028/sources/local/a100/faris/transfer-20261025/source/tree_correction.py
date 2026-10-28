"""Fixed shallow boosted offset correction; same 18 inputs and disjoint 6k labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,time
import joblib
import numpy as np
from scipy.special import expit,logit
from catboost import CatBoostClassifier,Pool
import baseline as b
from robustness import R,PERIODIC,inputs,check,read

PARAMS=dict(iterations=250,depth=3,learning_rate=.03,l2_leaf_reg=30,
            bootstrap_type='No',random_strength=0,thread_count=2,
            loss_function='Logloss',verbose=False,allow_writing_files=False)

def matrix(frame,columns,cats):
    x=b.prepare(frame,columns,cats)
    for c in cats:x[c]=x[c].astype(str)
    return x

def run(fold):
    check()
    dest=R/f'tree_correction/fold{fold}'
    dest.mkdir(parents=True,exist_ok=False)
    original,ix,q,context,yc,columns,raw,frame,mature,models=inputs(fold)
    head,sp=models['baseline']
    assert len(head['columns'])==18
    x=matrix(q,head['columns'],head['categorical'])
    spline=b.predict_spline(sp,frame.iloc[ix['validation']])
    start=time.time()
    checks=[]
    for mode in ['original','fresh_kumo_seed']:
        if mode=='original':p=raw
        else:
            root=R/f'robustness/fold{fold}'
            assert read(root/'complete.json')['status']=='complete'
            p=np.load(root/'raw_seed_repeat.npz')['prediction']
        offset=logit(np.clip(p,1e-6,1-1e-6))
        training=Pool(x.iloc[:6000],label=yc,cat_features=head['categorical'],baseline=offset[:6000])
        model=CatBoostClassifier(**PARAMS,random_seed=442025+fold)
        model.fit(training)
        # Verify installed CatBoost's baseline addition before using raw predictions.
        bare=Pool(x.iloc[:32],cat_features=head['categorical'])
        withbase=Pool(x.iloc[:32],cat_features=head['categorical'],baseline=offset[:32])
        difference=model.predict(withbase,prediction_type='RawFormulaVal')-model.predict(bare,prediction_type='RawFormulaVal')
        error=float(np.max(np.abs(difference-offset[:32])))
        assert error<1e-6,error
        path=dest/f'{mode}.cbm'
        model.save_model(path)
        pool=Pool(x.iloc[6000:],cat_features=head['categorical'])
        tree=model.predict(pool,prediction_type='RawFormulaVal')
        pred=.75*expit(offset[6000:]+.5*tree)+.25*spline
        replay=CatBoostClassifier();replay.load_model(path)
        assert np.array_equal(tree,replay.predict(pool,prediction_type='RawFormulaVal'))
        np.savez_compressed(dest/f'{mode}.npz',rows=ix['validation'],application_id=q.application_id.iloc[6000:].to_numpy(str),prediction=pred)
        checks.append(dict(mode=mode,model_sha256=b.sha(path),baseline_addition_error=error))
    b.dump(dict(status='complete',fold=fold,seconds=time.time()-start,params=PARAMS,
                columns=head['columns'],categorical=head['categorical'],training_rows=6000,
                validation_rows=14000,no_validation_labels_used_for_fit=True,checks=checks,
                source_sha256=b.sha(Path(__file__))),dest/'complete.json')

def report():
    z=np.load(PERIODIC/'OOF.npz');y,fold=z['target'],z['fold']
    pred={}
    for mode in ['original','fresh_kumo_seed']:
        a=np.full(len(y),np.nan)
        for f in range(5):
            root=R/f'tree_correction/fold{f}'
            assert read(root/'complete.json')['status']=='complete'
            p=np.load(root/f'{mode}.npz');assert np.array_equal(p['rows'],np.flatnonzero(fold==f))
            a[p['rows']]=p['prediction']
        assert np.isfinite(a).all();pred[mode]=a
    pred['two_seed_mean']=.5*(pred['original']+pred['fresh_kumo_seed'])
    metrics={}
    for mode,p in pred.items():
        m=b.scores(y,p);m['fold_KS']=[b.scores(y[fold==f],p[fold==f])['KS']for f in range(5)];metrics[mode]=m
    b.dump(dict(status='complete',metrics=metrics,params=PARAMS,interpretation='Single fixed tree structure, no outer-fold early stopping or parameter search. Reused development evidence.'),R/'TREE_CORRECTION_RESULT.json')
    np.savez_compressed(R/'TREE_CORRECTION_OOF.npz',target=y,fold=fold,**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','report']);p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args()
    if args.action=='run':run(args.fold)
    else:report()
