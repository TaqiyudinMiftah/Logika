"""H015: class-balanced Kumo context with a same-size stratified control."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import argparse,time,json
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from scipy.stats import ks_2samp
import baseline as b
import feature_head
from gpu_backend import Engine
from robustness import R,A,INPUT,PACKETS,inputs,stages,read
from joint_correction import weighted_ks

D=R/'balanced_context'
MODES=['original','fresh_kumo_seed']
KINDS=['balanced','proportional']


def check():
    plan=read(R/'H015.json')
    assert time.time()<plan['training_deadline_epoch'] and not(R/'STOP').exists()
    for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h,n
    assert sum(p.stat().st_size for p in R.rglob('*')if p.is_file())<600000000
    return plan


def prepare():
    plan=check();D.mkdir(exist_ok=False);checks=[]
    for f in range(5):
        source=PACKETS/f'fold{f}/context.parquet'
        assert b.sha(source)==read(source.parent/'manifest.json')['files']['context.parquet']
        context=pd.read_parquet(source,columns=['application_id','target'])
        labels=context.target.to_numpy(int);classes=[np.flatnonzero(labels==c)for c in (0,1)]
        assert len(context)==50000 and len(classes[1])<len(classes[0])
        n=2*len(classes[1]);pi=float(labels.mean());dest=D/f'fold{f}';dest.mkdir()
        indices=np.load(INPUT/f'artifacts/fold{f}/indices.npz')
        for mode in MODES:
            sample_seed=(2026102515 if mode=='original'else 3026102515)+f
            rng=np.random.default_rng(sample_seed)
            balanced=np.sort(np.r_[classes[1],rng.choice(classes[0],len(classes[1]),replace=False)])
            npositive=int(round(n*pi))
            proportional=np.sort(np.r_[rng.choice(classes[1],npositive,replace=False),rng.choice(classes[0],n-npositive,replace=False)])
            for kind,rows in [('balanced',balanced),('proportional',proportional)]:
                assert len(np.unique(rows))==n and not np.intersect1d(indices['context'][rows],np.r_[indices['correction'],indices['validation']]).size
                root=dest/f'{kind}__{mode}';root.mkdir()
                np.savez_compressed(root/'selected_rows.npz',context_local_rows=rows,original_rows=indices['context'][rows])
                metadata=dict(kind='indexed_original_packet_Kumo_context',context_source=str(source),context_source_sha256=b.sha(source),
                              context_local_rows=rows,original_rows=indices['context'][rows],selected_ID_hash=b.ids_hash(context,rows),
                              source_context_rows=50000,selected_rows=n,full_context_prevalence=pi,selected_prevalence=float(labels[rows].mean()),
                              sampling_seed=sample_seed,backbone_seed=(2026 if mode=='original'else 12026)+f,
                              columns=read(A/f'kumo-coverage-placement-20261021/conditions/fold{f}/control90/boundary.json')['feature_columns'],
                              checkpoint_sha256=plan['checkpoint_sha256'],checkpoint=str(A/'ranking-pipeline-20261020/assets/classifier.pt'),
                              num_estimators=8,feature_priors='unchanged original50000 context',prior_adjustment='logit(full_context_prevalence)-logit(selected_prevalence)')
                joblib.dump(metadata,root/'kumo_model.joblib',compress=3)
                checks.append(dict(fold=f,kind=kind,mode=mode,selected_rows=n,selected_positives=int(labels[rows].sum()),
                                   selection_file_sha256=b.sha(root/'selected_rows.npz'),canonical_sha256=b.sha(root/'kumo_model.joblib')))
    b.dump(dict(status='passed_for_fixed_H015',checks=checks,hypothesis_sha256=b.sha(R/'H015.json'),
                original_context_subset_only=True,zero_duplicate_selected_rows=True,correction_and_validation_disjoint=True,
                all_subset_IDs_frozen_before_any_new_Kumo_fit=True,no_validation_labels_used=True),D/'GATE.json')
    print(json.dumps(dict(status='prepared',contexts=len(checks))),flush=True)


def run(f):
    plan=check();gate=read(D/'GATE.json');assert gate['hypothesis_sha256']==b.sha(R/'H015.json')
    original,ix,q,context,yc,columns,raw,frame,mature,models=inputs(f)
    query,table=stages(q,frame,mature,'maturity');hspec,spspec=models['maturity']
    spline=b.predict_spline(spspec,table.iloc[ix['validation']])
    assert b.sha(A/'ranking-pipeline-20261020/assets/classifier.pt')==plan['checkpoint_sha256']
    engine=Engine(A/'ranking-pipeline-20261020/assets/classifier.pt',columns,context)
    for mode in MODES:
        for kind in KINDS:
            check();root=D/f'fold{f}/{kind}__{mode}';assert not(root/'complete.json').exists();started=time.time()
            canonical=joblib.load(root/'kumo_model.joblib');rows=canonical['context_local_rows']
            assert canonical['context_source_sha256']==b.sha(PACKETS/f'fold{f}/context.parquet')
            assert b.ids_hash(context,rows)==canonical['selected_ID_hash']
            modelprob=engine.fit_predict(rows,query,canonical['backbone_seed'],cache_path=None)
            assert engine.last_prediction_replay_error==0. and np.isfinite(modelprob).all()
            shift=float(logit(canonical['full_context_prevalence'])-logit(canonical['selected_prevalence']))
            adjusted=expit(logit(np.clip(modelprob.astype(float),1e-6,1-1e-6))+shift)
            # The analytic inverse prior shift is monotone; clipping can introduce ties.
            assert np.all(np.diff(adjusted[np.argsort(modelprob,kind='stable')])>=0)
            np.savez_compressed(root/'raw_predictions.npz',rows=np.r_[ix['correction'],ix['validation']],
                                application_id=query.application_id.to_numpy(str),model_probability=modelprob,
                                prior_adjusted_probability=adjusted,logit_prior_shift=shift)
            correction=query.iloc[:6000].copy();correction['target']=yc
            head=feature_head.fit(correction,adjusted[:6000],hspec['columns'],.003,.5)
            joblib.dump(head,root/'head.joblib',compress=3)
            pred=.75*b.corrected_probability(head,query.iloc[6000:],adjusted[6000:])[0]+.25*spline
            replay=.75*b.corrected_probability(joblib.load(root/'head.joblib'),query.iloc[6000:],adjusted[6000:])[0]+.25*spline
            assert np.array_equal(pred,replay)
            np.savez_compressed(root/'validation.npz',rows=ix['validation'],application_id=query.application_id.iloc[6000:].to_numpy(str),prediction=pred)
            b.dump(dict(status='complete',fold=f,kind=kind,mode=mode,seconds=time.time()-started,
                        selected_context_rows=len(rows),raw_replay_max_error=engine.last_prediction_replay_error,
                        spline_source=str(A/f'maturity-placement-fe-20261021/fold{f}/spline.joblib'),
                        spline_source_sha256=b.sha(A/f'maturity-placement-fe-20261021/fold{f}/spline.joblib'),
                        correction_rows=6000,validation_rows=14000,hypothesis_sha256=b.sha(R/'H015.json'),
                        files={p.name:b.sha(p)for p in root.iterdir()if p.is_file()}),root/'complete.json')
            print(json.dumps(dict(stage='context_complete',fold=f,kind=kind,mode=mode,seconds=time.time()-started)),flush=True)
    del engine


def report():
    assert not(D/'RESULT.json').exists()
    ref=np.load(R/'ROBUSTNESS_OOF.npz');y,fold=ref['target'],ref['fold']
    predictions={f'{kind}__{mode}':np.full(len(y),np.nan)for kind in KINDS for mode in MODES}
    for f in range(5):
        for key in predictions:
            root=D/f'fold{f}/{key}';complete=read(root/'complete.json');assert complete['status']=='complete'
            for n,h in complete['files'].items():assert b.sha(root/n)==h,n
            z=np.load(root/'validation.npz');assert np.array_equal(z['rows'],np.flatnonzero(fold==f))
            predictions[key][z['rows']]=z['prediction']
    metrics={}
    for key,p in predictions.items():
        assert np.isfinite(p).all();score=b.scores(y,p)
        assert abs(score['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-14
        score['fold_KS']=[b.scores(y[fold==f],p[fold==f])['KS']for f in range(5)];metrics[key]=score
    evaluated=dict(predictions)
    evaluated.update({f'full__{mode}':ref[f'maturity__{mode}']for mode in MODES})
    info={}
    for name,p in evaluated.items():
        order=np.argsort(p,kind='stable');s=p[order];info[name]=(order,y[order],np.r_[s[1:]!=s[:-1],True])
    groups=[np.flatnonzero((fold==f)&(y==cl))for f in range(5)for cl in (0,1)]
    rng=np.random.default_rng(20261025150);draws={f'{mode}__{control}':[]for mode in MODES for control in ['full','proportional']}
    for _ in range(2000):
        counts=np.zeros(len(y),dtype=int)
        for g in groups:counts[g]=rng.multinomial(len(g),np.full(len(g),1/len(g)))
        vals={name:weighted_ks(t,counts[o],ends)for name,(o,t,ends)in info.items()}
        for mode in MODES:
            for control in ['full','proportional']:draws[f'{mode}__{control}'].append(vals[f'balanced__{mode}']-vals[f'{control}__{mode}'])
    comparisons={}
    for mode in MODES:
        for control in ['full','proportional']:
            c=evaluated[f'{control}__{mode}'];p=predictions[f'balanced__{mode}'];key=f'{mode}__{control}'
            comparisons[key]=dict(KS_delta=b.scores(y,p)['KS']-b.scores(y,c)['KS'],
                                  four_contrast_adjusted95pct_interval=np.quantile(draws[key],[.00625,.99375]).tolist(),
                                  fold_wins=int(sum(b.scores(y[fold==f],p[fold==f])['KS']>b.scores(y[fold==f],c[fold==f])['KS']for f in range(5))))
    passed=(metrics['balanced__original']['KS']>.5314098055476294 and comparisons['original__full']['fold_wins']>=3
            and all(c['KS_delta']>0 and c['four_contrast_adjusted95pct_interval'][0]>0 for c in comparisons.values()))
    np.savez_compressed(D/'OOF.npz',target=y,fold=fold,**predictions);np.savez_compressed(D/'BOOTSTRAP.npz',**draws)
    out=dict(status='complete',metrics=metrics,comparisons=comparisons,promotion_passed=bool(passed),selected='balanced__original'if passed else None,
             target_054_reached=metrics['balanced__original']['KS']>=.54,hypothesis_sha256=b.sha(R/'H015.json'),
             independent_confirmation=False,qualification='Originalseed balanced primary; freshsampling+backboneseed replication; proportional contexts are mechanism controls only. Conditional bootstrap excludes prior adaptive selection and refit uncertainty.')
    b.dump(out,D/'RESULT.json');print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','report']);p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run(a.fold)
    else:report()
