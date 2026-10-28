"""Refit one frozen finalist using retained full-data contexts and exact raw inputs."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import argparse,json,time,copy
import numpy as np
import pandas as pd
import joblib
import baseline as b
import feature_head,spline_full
from maturity_features import MaturityFeatures
from gpu_backend import Engine
from robustness import R,A,INPUT,MAIN,MATURITY,PERIODIC,REPLACEMENT,read

FULL=A/'submission-full-periodicity-20261021'

def get_plan(training=True):
    plan=read(R/'FULL_REFIT_PLAN.json')
    assert time.time()<plan['training_deadline_epoch' if training else 'submission_deadline_epoch']
    for name,digest in plan['source_sha256'].items():
        assert b.sha(R/'source'/name)==digest,name
    assert b.sha(R/plan['selection_evidence']['path'])==plan['selection_evidence']['sha256']
    return plan

def refit(f):
    plan=get_plan()
    variant,averaging=plan['variant'],plan['averaging']
    dest=R/f'final/fold{f}'
    dest.mkdir(parents=True,exist_ok=False)
    start=time.time()
    oldroot=FULL/f'models/fold{f}'
    canonical=joblib.load(oldroot/'kumo_model.joblib')
    ix=np.load(FULL/f'input/artifacts/fold{f}/indices.npz')
    assert len(ix['context'])==64000 and len(ix['correction'])==6000
    feature_root=FULL/f'feature_build/fold{f}'
    assert b.sha(feature_root/'features.parquet')==read(feature_root/'complete.json')['feature_table_sha256']
    frame=pd.read_parquet(feature_root/'features.parquet')
    apps=pd.read_parquet(FULL/'applications.parquet')
    assert frame.application_id.tolist()==apps.application_id.tolist() and 'target'not in frame
    y=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['target']).target.to_numpy(int)
    assert canonical['context'].application_id.tolist()==frame.application_id.iloc[ix['context']].tolist()
    assert np.array_equal(canonical['context'].target.to_numpy(int),y[ix['context']])
    assert set(canonical['context'].application_id).isdisjoint(frame.application_id.iloc[ix['correction']])
    assert set(canonical['context'].application_id).isdisjoint(frame.application_id.iloc[70000:])
    query_rows=np.r_[ix['correction'],np.arange(70000,100000)]
    query=frame.iloc[query_rows][['application_id']+canonical['columns']].copy()
    old_raw=np.load(oldroot/'raw_predictions.npz')
    assert np.array_equal(old_raw['rows'],query_rows)
    assert np.array_equal(old_raw['application_id'],query.application_id.to_numpy(str))
    raw={'original':old_raw['prediction']}
    if averaging=='two_seed_mean':
        engine=Engine(A/'ranking-pipeline-20261020/assets/classifier.pt',canonical['columns'],canonical['context'])
        raw['fresh']=engine.fit_predict(np.arange(64000),query,12026+f,cache_path=None)
        assert engine.last_prediction_replay_error==0.
        del engine
        np.savez_compressed(dest/'raw_fresh.npz',rows=query_rows,application_id=query.application_id.to_numpy(str),prediction=raw['fresh'])
    if variant=='baseline':
        hspec=joblib.load(MAIN/f'conditions/fold{f}/control90/correction.joblib')
        sspec=joblib.load(MAIN/f'shared/fold{f}/spline.joblib')
    elif variant=='maturity':
        hspec=joblib.load(MATURITY/f'heads/matched_maturity/fold{f}/model.joblib')
        sspec=joblib.load(REPLACEMENT/f'fold{f}/spline.joblib')
    else:
        assert variant=='periodicity'
        hspec=joblib.load(PERIODIC/f'models/true/fold{f}/head.joblib')
        sspec=joblib.load(PERIODIC/f'models/true/fold{f}/spline.joblib')
    cov=pd.read_parquet(FULL/'coverage.parquet').set_index('application_id').loc[frame.application_id]
    spectral=pd.read_parquet(PERIODIC/'true_features.parquet').set_index('application_id').loc[frame.application_id]
    for c in hspec['columns']:
        if c in cov:frame[c]=cov[c].to_numpy()
        elif c in spectral:frame[c]=spectral[c].to_numpy()
    if variant!='baseline':
        history=pd.read_csv(A/'exploration-12h-20261017/raw/repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'])
        factory=MaturityFeatures.load(oldroot/'maturity_transformer.json')
        mature=factory.transform(apps,history).set_index('application_id').loc[frame.application_id]
        for c in ['re_late_1_posterior','re_late_30_posterior']:frame[c]=mature[c].to_numpy()
        factory.save(dest/'maturity_transformer.json')
    for c in sspec['categorical']:frame[c]=frame[c].astype(object)
    training=frame.iloc[:70000].copy()
    training['target']=y
    sp=spline_full.fit(training,sspec['columns'])
    assert sp['resolved_parameters']==sspec['resolved_parameters']
    joblib.dump(sp,dest/'spline.joblib',compress=3)
    test=frame.iloc[70000:].copy()
    selected=list(dict.fromkeys(hspec['columns']+sspec['columns']))
    test[['application_id']+selected].to_parquet(dest/'test_features.parquet',index=False)
    spred=b.predict_spline(sp,test)
    results=[]
    bundles=[]
    for mode,p in raw.items():
        modes=['ordinary'] if averaging!='three_head_mean' else ['ordinary','resample51025','resample61025']
        for draw in modes:
            if draw=='ordinary':rows=np.arange(6000)
            else:
                rng=np.random.default_rng(int(draw.replace('resample',''))+f)
                yc=y[ix['correction']]
                rows=np.concatenate([rng.choice(np.flatnonzero(yc==c),int((yc==c).sum()),replace=True)for c in (0,1)])
            selected_rows=ix['correction'][rows]
            headtrain=frame.iloc[selected_rows].copy()
            headtrain['target']=y[selected_rows]
            head=feature_head.fit(headtrain,p[:6000][rows],hspec['columns'],.003,.5)
            if plan.get('objective')=='direct_ks':
                from direct_ks import fit as fit_direct
                head=fit_direct(head,headtrain,p[:6000][rows])
            path=dest/f'head_{mode}_{draw}.joblib'
            joblib.dump(head,path,compress=3)
            prediction=.75*b.corrected_probability(head,test,p[6000:])[0]+.25*spred
            replay=.75*b.corrected_probability(joblib.load(path),test,p[6000:])[0]+.25*b.predict_spline(joblib.load(dest/'spline.joblib'),test)
            assert np.array_equal(prediction,replay)
            results.append(prediction)
            bundles.append(dict(head_file=path.name,raw_mode=mode,draw=draw))
    pred=np.mean(results,axis=0)
    assert np.isfinite(pred).all()and((pred>0)&(pred<1)).all()
    np.savez_compressed(dest/'test_predictions.npz',application_id=test.application_id.to_numpy(str),prediction=pred)
    b.dump(dict(status='complete',fold=f,variant=variant,averaging=averaging,seconds=time.time()-start,
                context_rows=64000,correction_rows=6000,spline_rows=70000,every_training_row_used=True,
                canonical_context_source=str(oldroot/'kumo_model.joblib'),canonical_context_sha256=b.sha(oldroot/'kumo_model.joblib'),
                original_raw_source=str(oldroot/'raw_predictions.npz'),original_raw_sha256=b.sha(oldroot/'raw_predictions.npz'),
                bundles=bundles,Kumo_seeds=[canonical['seed']]+([12026+f]if averaging=='two_seed_mean'else[]),source_sha256=plan['source_sha256'],
                files={p.name:b.sha(p)for p in dest.iterdir()if p.is_file()}),dest/'complete.json')
    print(json.dumps(dict(stage='full_refit_fold_complete',fold=f,seconds=time.time()-start)),flush=True)

def finish():
    plan=get_plan(training=False)
    sample=pd.read_csv(A/'exploration-12h-20261017/raw/sample_submission(3).csv')
    assert sample.application_id.is_unique and len(sample)==30000
    predictions=[]
    for f in range(5):
        dest=R/f'final/fold{f}'
        complete=read(dest/'complete.json')
        assert complete['status']=='complete'
        for name,digest in complete['files'].items():assert b.sha(dest/name)==digest,name
        a=np.load(dest/'test_predictions.npz')
        p=pd.Series(a['prediction'],index=a['application_id']).reindex(sample.application_id).to_numpy()
        assert np.isfinite(p).all()
        predictions.append(p)
    pred=np.mean(predictions,axis=0)
    path=R/plan['submission_filename']
    assert not path.exists()
    pd.DataFrame(dict(application_id=sample.application_id,default_probability=pred)).to_csv(path,index=False)
    readback=pd.read_csv(path)
    assert readback.columns.tolist()==['application_id','default_probability']
    assert readback.application_id.tolist()==sample.application_id.tolist()
    assert np.max(np.abs(readback.default_probability.to_numpy()-pred))<1e-15
    b.dump(dict(status='passed',submission=str(path),submission_sha256=b.sha(path),rows=30000,
                training_rows_per_run=70000,context_rows_per_run=64000,correction_rows_per_run=6000,
                runs=5,validation_score_of_full_refit=None,selection_evidence=plan['selection_evidence'],
                source_sha256=plan['source_sha256'],uploaded=False),R/'FULL_REFIT_DELIVERY.json')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['run','finish'])
    p.add_argument('--fold',type=int,choices=range(5))
    args=p.parse_args()
    if args.action=='run':refit(args.fold)
    else:finish()
