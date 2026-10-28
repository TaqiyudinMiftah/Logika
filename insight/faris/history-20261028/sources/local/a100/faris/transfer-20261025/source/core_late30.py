"""Frozen H007 ablation: change the 30-day core input while retaining recent 1-day evidence."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,time
import joblib
import numpy as np
import pandas as pd
import baseline as b
import feature_head
from gpu_backend import Engine
from robustness import R,A,PERIODIC,inputs,stages,read

REPLACE={'v3_repay_all_late_30_posterior_24':'matched_beta_late_30'}

def run(fold):
    plan=read(R/'H007.json')
    assert time.time()<plan['training_deadline_epoch']
    assert b.sha(Path(__file__))==plan['source_sha256']
    assert read(R/'GATE.json')['status']=='passed_for_bounded_paired_robustness_pilots'
    root=R/f'core_late30/fold{fold}'
    root.mkdir(parents=True,exist_ok=False)
    started=time.time()
    original,ix,q,context,yc,columns,oldraw,frame,mature,models=inputs(fold)
    headspec,spspec=models['maturity']
    query,table=stages(q,frame,mature,'maturity')
    before=context.copy()
    assert set(REPLACE).issubset(columns)
    for col,new in REPLACE.items():
        context[col]=mature.loc[context.application_id,new].to_numpy(float)
        query[col]=mature.loc[query.application_id,new].to_numpy(float)
    for col in columns:
        if col not in REPLACE:pd.testing.assert_series_equal(before[col],context[col],check_exact=True)
    assert context.application_id.tolist()==before.application_id.tolist()
    assert np.array_equal(context.target.to_numpy(),before.target.to_numpy())
    assert all(np.isfinite(context[col]).all()and context[col].between(0,1,inclusive='both').all()for col in REPLACE)
    transformer=read(A/f'repayment-severity-fe-20261021/fold{fold}/transformer.json')
    # This table was already checked against its context-only prior fit by inputs()
    # and the original feature-factory audit; preserve its exact source identity.
    assert b.sha(A/f'repayment-severity-fe-20261021/fold{fold}/features.parquet')==transformer['feature_hash']
    joblib.dump(dict(context=context[['application_id']+columns+['target']],columns=columns,
                     checkpoint_sha256=plan['checkpoint_sha256'],seeds=[2026+fold,12026+fold],
                     replacement=REPLACE,maturity_transformer_source=str(A/f'repayment-severity-fe-20261021/fold{fold}/transformer.json')),
                root/'canonical_context.joblib',compress=3)
    spline=b.predict_spline(spspec,table.iloc[ix['validation']])
    control=.75*b.corrected_probability(headspec,stages(q,frame,mature,'maturity')[0].iloc[6000:],oldraw[6000:])[0]+.25*spline
    expected=np.load(PERIODIC/'OOF.npz')['maturity'][ix['validation']]
    assert np.max(np.abs(control-expected))<2e-14
    engine=Engine(A/'ranking-pipeline-20261020/assets/classifier.pt',columns,context)
    checks=[]
    for label,seed in [('original',2026+fold),('fresh',12026+fold)]:
        raw=engine.fit_predict(np.arange(50000),query,seed,cache_path=None)
        assert engine.last_prediction_replay_error==0. and len(raw)==20000 and np.isfinite(raw).all()
        np.savez_compressed(root/f'{label}_raw.npz',rows=np.r_[ix['correction'],ix['validation']],application_id=query.application_id.to_numpy(str),prediction=raw)
        training=query.iloc[:6000].copy();training['target']=yc
        head=feature_head.fit(training,raw[:6000],headspec['columns'],.003,.5)
        path=root/f'{label}_head.joblib';joblib.dump(head,path,compress=3)
        pred=.75*b.corrected_probability(head,query.iloc[6000:],raw[6000:])[0]+.25*spline
        replay=.75*b.corrected_probability(joblib.load(path),query.iloc[6000:],raw[6000:])[0]+.25*spline
        assert np.array_equal(pred,replay)
        np.savez_compressed(root/f'{label}_validation.npz',rows=ix['validation'],application_id=query.application_id.iloc[6000:].to_numpy(str),prediction=pred)
        checks.append(dict(mode=label,seed=seed,head_sha256=b.sha(path),Kumo_first_batch_replay_error=engine.last_prediction_replay_error))
        print(dict(stage='core_late30_seed_complete',fold=fold,mode=label,seed=seed),flush=True)
    del engine
    b.dump(dict(status='complete',fold=fold,seconds=time.time()-started,replacement=REPLACE,
                context_rows=50000,correction_rows=6000,validation_rows=14000,head_inputs=18,spline_inputs=77,
                prior_source_sha256=b.sha(A/f'repayment-severity-fe-20261021/fold{fold}/transformer.json'),
                query_has_no_default_labels=True,no_new_features_or_future_fields=True,
                checks=checks,source_sha256=plan['source_sha256'],
                files={p.name:b.sha(p)for p in root.iterdir()if p.is_file()}),root/'complete.json')

def report():
    control=np.load(R/'ROBUSTNESS_OOF.npz');y,fold=control['target'],control['fold']
    predictions={}
    for mode in ['original','fresh']:
        pred=np.full(len(y),np.nan)
        for f in range(5):
            root=R/f'core_late30/fold{f}'
            assert read(root/'complete.json')['status']=='complete'
            a=np.load(root/f'{mode}_validation.npz')
            assert np.array_equal(a['rows'],np.flatnonzero(fold==f))
            pred[a['rows']]=a['prediction']
        assert np.isfinite(pred).all();predictions[mode]=pred
    predictions['two_seed_mean']=.5*(predictions['original']+predictions['fresh'])
    reference={'original':control['maturity__original'],'fresh':control['maturity__fresh_kumo_seed'],'two_seed_mean':control['maturity__two_seed_mean']}
    metrics={};effects={}
    for mode,pred in predictions.items():
        score=b.scores(y,pred);score['fold_KS']=[b.scores(y[fold==f],pred[fold==f])['KS']for f in range(5)]
        ref=b.scores(y,reference[mode]);ref['fold_KS']=[b.scores(y[fold==f],reference[mode][fold==f])['KS']for f in range(5)]
        diff=np.array(score['fold_KS'])-ref['fold_KS'];metrics[mode]=dict(candidate=score,control=ref)
        effects[mode]=dict(KS_delta=score['KS']-ref['KS'],AUC_delta=score['AUC']-ref['AUC'],fold_KS_delta=diff.tolist(),fold_wins=int(sum(diff>0)))
    plan=read(R/'H007.json')
    accepted=all(effects[m]['KS_delta']>0 for m in effects)and metrics['two_seed_mean']['candidate']['KS']>plan['current_best_local_KS']and effects['two_seed_mean']['fold_wins']>=3
    b.dump(dict(status='complete',metrics=metrics,effects=effects,promotion_predicate_passed=accepted,
                target_054_reached=metrics['two_seed_mean']['candidate']['KS']>=.54,
                hypothesis_sha256=b.sha(R/'H007.json'),independent_confirmation=False),R/'CORE_LATE30_RESULT.json')
    np.savez_compressed(R/'CORE_LATE30_OOF.npz',target=y,fold=fold,**predictions)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','report']);p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args()
    if args.action=='run':run(args.fold)
    else:report()
