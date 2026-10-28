"""Conditional finalist D: full-data refit of the selected frozen core intervention."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,time,shutil
import numpy as np
import pandas as pd
import joblib
import baseline as b
import feature_head
from gpu_backend import Engine
from maturity_features import MaturityFeatures
from robustness import R,A,INPUT,read

FULL=A/'submission-full-periodicity-20261021'
DEST=R/'candidate_D'

def run(f):
    plan=read(DEST/'FULL_REFIT_PLAN.json')
    assert time.time()<plan['training_deadline_epoch']
    tag={'H006':'CORE_MATURITY','H007':'CORE_LATE30'}[plan['core_hypothesis']]
    replacement_spec=read(R/f'{plan["core_hypothesis"]}.json')['replacement']
    assert replacement_spec==plan['replacement']
    assert read(R/f'{tag}_RESULT.json')['promotion_predicate_passed']is True
    assert b.sha(R/f'{tag}_RESULT.json')==plan['selection_evidence']['sha256']
    assert read(R/f'{tag}_BOUNDARY_AUDIT.json')['all_five_contexts_verified'] is True
    assert b.sha(A/'ranking-pipeline-20261020/assets/classifier.pt')==plan['checkpoint_sha256']
    for name,digest in plan['source_sha256'].items():assert b.sha(R/'source'/name)==digest,name
    root=DEST/f'final/fold{f}';root.mkdir(parents=True,exist_ok=False)
    started=time.time()
    old=FULL/f'models/fold{f}';source_B=R/f'candidate_B/final/fold{f}'
    canonical=joblib.load(old/'kumo_model.joblib')
    assert canonical['checkpoint_sha256']==plan['checkpoint_sha256']
    roles=np.load(FULL/f'input/artifacts/fold{f}/indices.npz')
    assert len(roles['context'])==64000 and len(roles['correction'])==6000
    feature=FULL/f'feature_build/fold{f}'
    assert b.sha(feature/'features.parquet')==read(feature/'complete.json')['feature_table_sha256']
    frame=pd.read_parquet(feature/'features.parquet')
    apps=pd.read_parquet(FULL/'applications.parquet')
    assert frame.application_id.tolist()==apps.application_id.tolist()and'target'not in frame
    y=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['target']).target.to_numpy(int)
    assert canonical['context'].application_id.tolist()==frame.application_id.iloc[roles['context']].tolist()
    assert np.array_equal(canonical['context'].target.to_numpy(int),y[roles['context']])
    history=pd.read_csv(A/'exploration-12h-20261017/raw/repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'])
    factory=MaturityFeatures.load(source_B/'maturity_transformer.json')
    mature=factory.transform(apps,history).set_index('application_id').loc[frame.application_id]
    factory.save(root/'maturity_transformer.json')
    rows=np.r_[roles['correction'],np.arange(70000,100000)]
    query=frame.iloc[rows][['application_id']+canonical['columns']].copy()
    context=canonical['context'].copy()
    # Kumo-only input replacement. The downstream spline keeps its inherited
    # v3 fields; its separate re_late fields receive the existing maturity change.
    for col,source in replacement_spec.items():
        threshold=1 if source.endswith('_1') else 30
        replacement=mature[f're_late_{threshold}_posterior']
        context[col]=replacement.loc[context.application_id].to_numpy()
        query[col]=replacement.loc[query.application_id].to_numpy()
    joblib.dump(dict(context=context,columns=canonical['columns'],seed=2026+f,
                     seeds=[2026+f,12026+f],num_estimators=8,
                     checkpoint_sha256=canonical['checkpoint_sha256'],replacement=replacement_spec,
                     source_context_sha256=b.sha(old/'kumo_model.joblib')),
                root/'kumo_model.joblib',compress=3)
    sp=joblib.load(source_B/'spline.joblib')
    assert sp['fit_rows']==70000
    headspec=joblib.load(source_B/'head_original.joblib')
    cov=pd.read_parquet(FULL/'coverage.parquet').set_index('application_id').loc[frame.application_id]
    for col in headspec['columns']:
        if col in cov:frame[col]=cov[col].to_numpy()
    for col in ['re_late_1_posterior','re_late_30_posterior']:frame[col]=mature[col].to_numpy()
    for col in sp['categorical']:frame[col]=frame[col].astype(object)
    test=frame.iloc[70000:]
    oldtest=pd.read_parquet(source_B/'test_features.parquet')
    pd.testing.assert_frame_equal(test[oldtest.columns].reset_index(drop=True),oldtest.reset_index(drop=True),check_exact=True)
    shutil.copy2(source_B/'spline.joblib',root/'spline.joblib')
    test[oldtest.columns].to_parquet(root/'test_features.parquet',index=False)
    spline=b.predict_spline(sp,test)
    engine=Engine(A/'ranking-pipeline-20261020/assets/classifier.pt',canonical['columns'],context)
    predictions=[];bundles=[]
    for mode,seed in [('original',2026+f),('fresh',12026+f)]:
        assert time.time()<plan['training_deadline_epoch']
        raw=engine.fit_predict(np.arange(64000),query,seed,cache_path=None)
        assert engine.last_prediction_replay_error==0. and np.isfinite(raw).all()
        np.savez_compressed(root/f'raw_{mode}.npz',rows=rows,application_id=query.application_id.to_numpy(str),prediction=raw)
        correction=frame.iloc[roles['correction']].copy();correction['target']=y[roles['correction']]
        head=feature_head.fit(correction,raw[:6000],headspec['columns'],.003,.5)
        model=root/f'head_{mode}.joblib';joblib.dump(head,model,compress=3)
        pred=.75*b.corrected_probability(head,test,raw[6000:])[0]+.25*spline
        replay=.75*b.corrected_probability(joblib.load(model),test,raw[6000:])[0]+.25*spline
        assert np.array_equal(pred,replay)
        predictions.append(pred);bundles.append(dict(head_file=model.name,raw_mode=mode,draw='ordinary'))
        print(dict(stage='core_full_refit_seed_complete',fold=f,mode=mode,seed=seed),flush=True)
    del engine
    prediction=np.mean(predictions,axis=0)
    np.savez_compressed(root/'test_predictions.npz',application_id=test.application_id.to_numpy(str),prediction=prediction)
    b.dump(dict(status='complete',fold=f,variant='maturity',averaging='two_seed_mean',seconds=time.time()-started,
                context_rows=64000,correction_rows=6000,spline_rows=70000,
                canonical_context_source=str(root/'kumo_model.joblib'),canonical_context_sha256=b.sha(root/'kumo_model.joblib'),
                original_raw_source=str(root/'raw_original.npz'),original_raw_sha256=b.sha(root/'raw_original.npz'),
                spline_source=str(source_B/'spline.joblib'),spline_source_sha256=b.sha(source_B/'spline.joblib'),
                downstream_feature_values_exactly_match_B=True,replacement=replacement_spec,bundles=bundles,
                source_sha256=plan['source_sha256'],files={p.name:b.sha(p)for p in root.iterdir()if p.is_file()}),root/'complete.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);args=p.parse_args();run(args.fold)
