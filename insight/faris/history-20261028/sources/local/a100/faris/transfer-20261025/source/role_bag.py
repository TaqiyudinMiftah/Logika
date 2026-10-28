"""H010: match validation to a five-context submission, with train-only priors."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,time,json,subprocess,sys,shutil
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
import baseline as b
import feature_head,spline_rolebag
from direct_ks import fit as direct_fit
from maturity_features import MaturityFeatures
from gpu_backend import Engine
from robustness import R,A,INPUT,MAIN,read

DEST=R/'role_bag'
SCRATCH=Path('/home/faris/logika-transfer-rolebag-20261025/cohort')
FULL=A/'submission-full-periodicity-20261021'
PY='/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python'
COVER=['risk_conditional_negative_months','risk_conditional_nonnegative_months','aux_account_months','aux_orders90','aux_ecommerce_observed_days','aux_allocation_count']

def check():
    plan=read(R/'H010.json')
    assert time.time()<plan['CV_deadline_epoch']
    assert not(R/'STOP').exists()
    for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h,n
    return plan

def init():
    plan=check();assert not DEST.exists();DEST.mkdir()
    assert not SCRATCH.exists();(SCRATCH/'source').mkdir(parents=True)
    (SCRATCH/'vendor').symlink_to(A/'fold-local-priors-20261021/vendor',target_is_directory=True)
    shutil.copy2(R/'source/cohort_rolebag.py',SCRATCH/'source/build.py')
    inp=SCRATCH/'input';(inp/'data').mkdir(parents=True)
    (inp/'data/train_features.parquet').symlink_to(INPUT/'data/train_features.parquet')
    shutil.copy2(INPUT/'BASELINE_LOCK.json',inp/'BASELINE_LOCK.json')
    feature_plan=read(A/'fold-local-priors-20261021/plan.json')
    feature_plan.update(id='H010-train56k-cohort-priors',original_deadline=plan['CV_deadline_epoch'],input=str(inp),shared_fitted_statistics='All56k outer-training covariates; no outer-validation or test covariates fit shared statistics.')
    b.dump(feature_plan,SCRATCH/'plan.json')
    train=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','target'])
    checks=[]
    for outer in range(5):
        old=np.load(INPUT/f'artifacts/fold{outer}/indices.npz');tr=old['train'];val=old['validation']
        assert len(tr)==56000 and len(val)==14000 and not np.intersect1d(tr,val).size
        out=inp/f'artifacts/fold{outer}';out.mkdir(parents=True)
        np.savez_compressed(out/'indices.npz',context=tr,validation=val,train=tr,correction=old['correction'])
        training_y=train.target.iloc[tr].to_numpy(int)
        for member in range(5):
            if member==outer:correction=old['correction']
            else:
                original=np.load(INPUT/f'artifacts/fold{member}/indices.npz')['correction']
                rng=np.random.default_rng(202610250+100*outer+member)
                selected=[];positive=int(round(6000*training_y.mean()))
                for cl,quota in [(0,6000-positive),(1,positive)]:
                    eligible=tr[training_y==cl];common=np.intersect1d(original,eligible)
                    kept=common if len(common)<=quota else rng.choice(common,quota,replace=False)
                    extra=rng.choice(np.setdiff1d(eligible,kept),quota-len(kept),replace=False)
                    selected.extend(np.r_[kept,extra].tolist())
                correction=np.sort(selected)
            context=np.setdiff1d(tr,correction)
            assert len(context)==50000 and len(correction)==6000
            assert not np.intersect1d(context,correction).size
            assert not np.intersect1d(np.r_[context,correction],val).size
            assert np.array_equal(np.sort(np.r_[context,correction]),np.sort(tr))
            out=DEST/f'outer{outer}/roles';out.mkdir(parents=True,exist_ok=True)
            path=out/f'member{member}.npz'
            np.savez_compressed(path,context=context,correction=correction,train=tr,validation=val)
            checks.append(dict(outer=outer,member=member,roles_file=str(path),roles_sha256=b.sha(path),context_rows=50000,correction_rows=6000,validation_rows=14000,no_validation_labels_used_for_sampling=True,training_union_verified=True))
    b.dump(dict(status='passed',checks=checks,checkpoint_sha256=b.sha(A/'ranking-pipeline-20261020/assets/classifier.pt'),new_preprocessing_scope='56k outer-training covariates',source_sha256=plan['source_sha256']),DEST/'ROLE_BOUNDARIES.json')

def prepare():
    plan=check()
    apps=pd.read_parquet(FULL/'applications.parquet')
    history=pd.read_csv(A/'exploration-12h-20261017/raw/repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'])
    train=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','target'])
    coverage=pd.read_parquet(FULL/'coverage.parquet').set_index('application_id')
    assert b.sha(SCRATCH/'source/build.py')==plan['source_sha256']['cohort_rolebag.py']
    for outer in range(5):
        check();started=time.time()
        subprocess.run([PY,str(SCRATCH/'source/build.py'),'--fold',str(outer)],check=True)
        source=SCRATCH/f'fold{outer}';complete=read(source/'complete.json')
        ix=np.load(INPUT/f'artifacts/fold{outer}/indices.npz');tr=ix['train'];val=ix['validation']
        assert complete['context_rows']==56000 and complete['validation_rows']==14000
        assert complete['fit_ids_sha256']==b.ids_hash(train,tr)
        assert complete['no_target_labels_available_to_feature_estimators'] is True
        assert b.sha(source/'features.parquet')==complete['feature_table_sha256']
        frame=pd.read_parquet(source/'features.parquet')
        assert frame.application_id.tolist()==apps.application_id.tolist() and 'target' not in frame
        headspec=joblib.load(A/f'repayment-severity-fe-20261021/heads/matched_maturity/fold{outer}/model.joblib')
        spspec=joblib.load(A/f'maturity-placement-fe-20261021/fold{outer}/spline.joblib')
        columns=read(MAIN/f'conditions/fold{outer}/control90/boundary.json')['feature_columns']
        for c in COVER:frame[c]=coverage.loc[frame.application_id,c].to_numpy()
        factory=MaturityFeatures().fit(apps,history,apps.application_id.iloc[tr])
        assert factory.parameters['fit_rows']==56000 and factory.parameters['default_labels_used'] is False
        mature=factory.transform(apps,history).set_index('application_id').loc[frame.application_id]
        for c in ['re_late_1_posterior','re_late_30_posterior']:frame[c]=mature[c].to_numpy()
        for c in spspec['categorical']:frame[c]=frame[c].astype(object)
        needed=list(dict.fromkeys(['application_id']+columns+headspec['columns']+spspec['columns']))
        frame=frame.iloc[:70000][needed].copy()
        assert frame.application_id.tolist()==train.application_id.tolist()
        out=DEST/f'outer{outer}/shared';out.mkdir(parents=True)
        temp=source/'required70k.parquet';frame.to_parquet(temp,index=False,compression='zstd',compression_level=9)
        current=sum(p.stat().st_size for p in R.rglob('*')if p.is_file()and not p.is_symlink())
        assert current+temp.stat().st_size<plan['archive_limit_bytes']
        assert shutil.disk_usage(DEST).free-temp.stat().st_size>600000000
        shutil.copy2(temp,out/'features.parquet');factory.save(out/'maturity_transformer.json')
        shutil.copy2(source/'prior_audit.json',out/'prior_audit.json');shutil.copy2(source/'complete.json',out/'COHORT_BUILD_COMPLETE.json')
        training=frame.iloc[tr].copy();training['target']=train.target.iloc[tr].to_numpy(int)
        sp=spline_rolebag.fit(training,spspec['columns']);joblib.dump(sp,out/'spline.joblib',compress=3)
        sv=b.predict_spline(sp,frame.iloc[val]);np.savez_compressed(out/'spline_validation.npz',rows=val,application_id=frame.application_id.iloc[val].to_numpy(str),prediction=sv)
        b.dump(dict(status='passed',outer=outer,preprocessing_fit_rows=56000,preprocessing_ID_hash=b.ids_hash(train,tr),spline_fit_rows=56000,validation_rows=14000,Kumo_columns=columns,head_columns=headspec['columns'],spline_columns=spspec['columns'],feature_table_sha256=b.sha(out/'features.parquet'),no_validation_or_test_covariates_fit_priors=True,real_default_labels_unavailable_to_cohort_feature_builder=True,seconds=time.time()-started,source_sha256=plan['source_sha256']),out/'PREPARED.json')
        if outer==0:
            b.dump(dict(status='passed_for_H010_role_ensemble_pilot',scope='Frozen25 contexts; training-only56k shared priors; no general search',roles_sha256=b.sha(DEST/'ROLE_BOUNDARIES.json'),source_sha256=plan['source_sha256'],snapshot_limitation_inherited=True,independent_generalization=False),DEST/'GATE.json')
        print(dict(stage='rolebag_shared_prepared',outer=outer,seconds=time.time()-started),flush=True)

def run(outer,member):
    plan=check();assert read(DEST/'GATE.json')['status']=='passed_for_H010_role_ensemble_pilot'
    shared=DEST/f'outer{outer}/shared';prep=read(shared/'PREPARED.json')
    assert b.sha(shared/'features.parquet')==prep['feature_table_sha256']
    boundary=next(x for x in read(DEST/'ROLE_BOUNDARIES.json')['checks']if x['outer']==outer and x['member']==member)
    assert b.sha(boundary['roles_file'])==boundary['roles_sha256']
    ix=np.load(boundary['roles_file']);frame=pd.read_parquet(shared/'features.parquet')
    train=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','target']);y=train.target.to_numpy(int)
    assert frame.application_id.tolist()==train.application_id.tolist()
    root=DEST/f'outer{outer}/member{member}';root.mkdir(parents=True,exist_ok=False);started=time.time()
    columns=prep['Kumo_columns'];context=frame.iloc[ix['context']][['application_id']+columns].copy();context['target']=y[ix['context']]
    assert not set(context.application_id)&set(frame.application_id.iloc[np.r_[ix['correction'],ix['validation']]])
    canonical=dict(kind='indexed_canonical_Kumo_context',feature_table=str(shared/'features.parquet'),feature_table_sha256=prep['feature_table_sha256'],context_indices=ix['context'],target_source=str(INPUT/'data/train_features.parquet'),target_source_sha256=b.sha(INPUT/'data/train_features.parquet'),context_ID_hash=b.ids_hash(frame,ix['context']),columns=columns,seed=2026+member,num_estimators=8,checkpoint_sha256=read(R/'H006.json')['checkpoint_sha256'])
    joblib.dump(canonical,root/'kumo_model.joblib',compress=3)
    rows=np.r_[ix['correction'],ix['validation']];query=frame.iloc[rows][['application_id']+columns].copy()
    engine=Engine(A/'ranking-pipeline-20261020/assets/classifier.pt',columns,context)
    raw=engine.fit_predict(np.arange(50000),query,2026+member,cache_path=None)
    assert engine.last_prediction_replay_error==0. and np.isfinite(raw).all();del engine
    np.savez_compressed(root/'raw_predictions.npz',rows=rows,application_id=query.application_id.to_numpy(str),prediction=raw)
    training=frame.iloc[ix['correction']].copy();training['target']=y[ix['correction']]
    anchor=feature_head.fit(training,raw[:6000],prep['head_columns'],.003,.5)
    sv=np.load(shared/'spline_validation.npz');assert np.array_equal(sv['rows'],ix['validation'])
    for mode,head in [('ordinary',anchor),('direct_KS',direct_fit(anchor,training,raw[:6000]))]:
        file=root/f'{mode}_head.joblib';joblib.dump(head,file,compress=3)
        pred=.75*b.corrected_probability(head,frame.iloc[ix['validation']],raw[6000:])[0]+.25*sv['prediction']
        replay=.75*b.corrected_probability(joblib.load(file),frame.iloc[ix['validation']],raw[6000:])[0]+.25*sv['prediction']
        assert np.array_equal(pred,replay)
        np.savez_compressed(root/f'{mode}_validation.npz',rows=ix['validation'],application_id=frame.application_id.iloc[ix['validation']].to_numpy(str),prediction=pred)
    b.dump(dict(status='complete',outer=outer,member=member,seconds=time.time()-started,Kumo_context_rows=50000,correction_rows=6000,shared_prior_rows=56000,validation_rows=14000,roles_sha256=boundary['roles_sha256'],checkpoint_sha256=canonical['checkpoint_sha256'],source_sha256=plan['source_sha256'],files={p.name:b.sha(p)for p in root.iterdir()if p.is_file()}),root/'complete.json')
    print(dict(stage='rolebag_member_complete',outer=outer,member=member,seconds=time.time()-started),flush=True)

def worker(gpu):
    jobs=[(o,j)for o in range(5)for j in range(5)][gpu::2]
    for outer,member in jobs:
        while not(DEST/f'outer{outer}/shared/PREPARED.json').exists():check();time.sleep(10)
        run(outer,member)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','prepare','run','worker']);p.add_argument('--outer',type=int);p.add_argument('--member',type=int);p.add_argument('--gpu',type=int,choices=[0,1]);a=p.parse_args()
    if a.action=='init':init()
    elif a.action=='prepare':prepare()
    elif a.action=='worker':worker(a.gpu)
    else:run(a.outer,a.member)
