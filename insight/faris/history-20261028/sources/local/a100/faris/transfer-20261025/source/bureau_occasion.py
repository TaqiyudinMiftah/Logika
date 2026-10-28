"""H014: finish the previously screened shared-application bureau hypothesis."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import argparse,time,json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
import baseline as b
import feature_head,spline_rolebag
import occasion_measurement as m
from robustness import R,A,INPUT,inputs,stages,read
from joint_correction import weighted_ks

D=R/'bureau_occasion'
RAW=A/'exploration-12h-20261017/raw'
EXTRA=['re_bureau_score_posterior','re_bureau_posterior_sd']
MODES=['original','fresh_kumo_seed']


def check():
    assert time.time()<read(R/'CONTINUATION_20261025.json')['training_deadline_epoch']
    assert not(R/'STOP').exists()
    for n,h in read(R/'H014.json')['source_sha256'].items():assert b.sha(R/'source'/n)==h,n
    assert sum(p.stat().st_size for p in R.rglob('*')if p.is_file())<600000000


def eligible(frame,apps):
    assert 'target'not in frame and'target'not in apps
    cut=frame.customer_id.map(apps.set_index('customer_id').decision_datetime)
    z=frame[(frame.report_datetime<cut)&(frame.available_datetime<cut)&frame.score.notna()].copy()
    z['day']=z.report_datetime.astype('int64')/86400e9
    return z


def build():
    check();D.mkdir(exist_ok=False);started=time.time()
    screen=A/'bureau-noise-scale-audit-20261021/SCREEN_RESULT.json'
    assert read(screen)['measurement_screen_passed'] is True
    assert b.sha(screen)==read(R/'H014.json')['measurement_screen_sha256']
    m.verify_identity(None)
    apps=pd.read_parquet(A/'submission-full-periodicity-20261021/applications.parquet',columns=['application_id','customer_id','decision_datetime'])
    train_ids=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id'])
    assert len(apps)==100000 and apps.customer_id.is_unique and apps.application_id.is_unique
    assert apps.application_id.iloc[:70000].tolist()==train_ids.application_id.tolist()
    raw=pd.read_csv(RAW/'bureau_observations.csv',parse_dates=['report_datetime','available_datetime'])
    data=eligible(raw,apps)
    assert not data.duplicated(['customer_id','application_id','bureau_provider']).any()
    receipts=[]
    for f in range(5):
        check();ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz')
        context=set(apps.customer_id.iloc[ix['context']]);assert len(context)==50000
        source=data[data.customer_id.isin(context)].copy()
        parameters={name:m.fit(source,flag)for name,flag in [('static',False),('occasion',True)]}
        path=A/f'fold-local-priors-20261021/fold{f}/features.parquet'
        assert b.sha(path)==read(path.parent/'complete.json')['feature_table_sha256']
        ref=pd.read_parquet(path,columns=['application_id']+EXTRA).set_index('application_id').loc[apps.application_id]
        static=m.transform(data,apps,parameters['static'])
        error={c:float(np.max(abs(static[i]-ref[c].to_numpy(float))))for i,c in enumerate(EXTRA)}
        assert max(error.values())<1e-8,error
        mean,sd=m.transform(data,apps,parameters['occasion'])
        # Own-history isolation, input-order invariance, and serialized replay.
        take=np.arange(0,100000,101);sample=apps.iloc[take].copy()
        sub=data[data.customer_id.isin(sample.customer_id)]
        sm,ss=m.transform(sub.sample(frac=1,random_state=2026102514),sample,parameters['occasion'])
        assert np.max(abs(sm-mean[take]))<1e-10 and np.max(abs(ss-sd[take]))<1e-10
        injected=raw.iloc[[0]].copy()
        injected['report_datetime']=injected.customer_id.map(apps.set_index('customer_id').decision_datetime)+pd.Timedelta(days=1)
        injected['available_datetime']=injected.report_datetime
        future=eligible(pd.concat([raw,injected],ignore_index=True),apps)
        pd.testing.assert_frame_equal(data.reset_index(drop=True),future.reset_index(drop=True))
        changed=data.copy();outside=~changed.customer_id.isin(context);changed.loc[outside,'score']+=100
        assert m.fit(changed[~outside],True)==parameters['occasion']
        root=D/f'features/fold{f}';root.mkdir(parents=True)
        out=pd.DataFrame({'application_id':apps.application_id,EXTRA[0]:mean,EXTRA[1]:sd})
        out.to_parquet(root/'features.parquet',index=False,compression='zstd',compression_level=9)
        b.dump(parameters,root/'parameters.json')
        again=m.transform(data,apps,read(root/'parameters.json')['occasion'])
        assert np.array_equal(again[0],mean)and np.array_equal(again[1],sd)
        receipt=dict(status='passed',fold=f,feature_sha256=b.sha(root/'features.parquet'),parameters_sha256=b.sha(root/'parameters.json'),
                     static_reproduction_errors=error,context_rows=50000,context_ID_sha256=b.ids_hash(apps,ix['context']),
                     default_labels_opened=False,noncontext_parameter_invariance=True,own_history_and_order_invariance=True,
                     future_event_invariance=True,serialized_transform_exact=True,occasion_variance=parameters['occasion']['occasion'])
        b.dump(receipt,root/'complete.json');receipts.append(receipt)
        print(json.dumps(dict(stage='feature_gate_fold',fold=f,static_error=max(error.values()),occasion_variance=receipt['occasion_variance'])),flush=True)
    b.dump(dict(status='passed_for_fixed_H014',default_labels_opened=False,folds=receipts,Gaussian_identity=True,
                raw_source_sha256=b.sha(RAW/'bureau_observations.csv'),measurement_screen_sha256=b.sha(screen),
                hypothesis_sha256=b.sha(R/'H014.json'),seconds=time.time()-started,
                limitations='Method and role checks only; default label reuse and inherited repayment snapshot limits remain.'),D/'FEATURE_GATE.json')


def run(f):
    check();gate=read(D/'FEATURE_GATE.json');assert gate['status']=='passed_for_fixed_H014'
    assert gate['hypothesis_sha256']==b.sha(R/'H014.json')
    dest=D/f'models/fold{f}';dest.mkdir(parents=True,exist_ok=False);started=time.time()
    original,ix,q,context,yc,columns,raw,frame,mature,models=inputs(f)
    query,table=stages(q,frame,mature,'maturity')
    source=D/f'features/fold{f}';receipt=read(source/'complete.json')
    assert b.sha(source/'features.parquet')==receipt['feature_sha256']
    extra=pd.read_parquet(source/'features.parquet').set_index('application_id')
    hspec,spspec=models['maturity']
    assert all(c in hspec['columns']and c in spspec['columns']for c in EXTRA)
    for c in EXTRA:
        query[c]=extra.loc[query.application_id,c].to_numpy(float)
        table[c]=extra.loc[table.application_id,c].to_numpy(float)
    training=table.iloc[ix['train']].copy();training['target']=original.target.iloc[ix['train']].to_numpy(int)
    sp=spline_rolebag.fit(training,spspec['columns']);joblib.dump(sp,dest/'spline.joblib',compress=3)
    assert sp['resolved_parameters']==spspec['resolved_parameters']
    spline=b.predict_spline(sp,table.iloc[ix['validation']])
    assert np.array_equal(spline,b.predict_spline(joblib.load(dest/'spline.joblib'),table.iloc[ix['validation']]))
    correction=query.iloc[:6000].copy();correction['target']=yc
    out=dict(rows=ix['validation'],application_id=query.application_id.iloc[6000:].to_numpy(str))
    for mode in MODES:
        base=raw if mode=='original'else np.load(R/f'robustness/fold{f}/raw_seed_repeat.npz')['prediction']
        head=feature_head.fit(correction,base[:6000],hspec['columns'],.003,.5)
        head['H014_roles']=dict(head_rows=6000,context_prior_rows=50000,spline_rows=56000,
                                correction_ID_hash=b.ids_hash(original,ix['correction']),feature_sha256=receipt['feature_sha256'])
        path=dest/f'head_{mode}.joblib';joblib.dump(head,path,compress=3)
        pred=.75*b.corrected_probability(head,query.iloc[6000:],base[6000:])[0]+.25*spline
        replay=.75*b.corrected_probability(joblib.load(path),query.iloc[6000:],base[6000:])[0]+.25*spline
        assert np.array_equal(pred,replay);out[mode]=pred
    np.savez_compressed(dest/'predictions.npz',**out)
    b.dump(dict(status='complete',fold=f,seconds=time.time()-started,hypothesis_sha256=b.sha(R/'H014.json'),
                feature_gate_sha256=b.sha(D/'FEATURE_GATE.json'),models_replayed_exactly=3,
                fitted_validation_rows=0,files={p.name:b.sha(p)for p in dest.iterdir()if p.is_file()}),dest/'complete.json')
    print(json.dumps(dict(stage='model_fold_complete',fold=f,seconds=time.time()-started)),flush=True)


def report():
    assert not(D/'RESULT.json').exists()
    ref=np.load(R/'ROBUSTNESS_OOF.npz');y,fold=ref['target'],ref['fold'];pred={k:np.full(len(y),np.nan)for k in MODES}
    for f in range(5):
        root=D/f'models/fold{f}';c=read(root/'complete.json');assert c['status']=='complete'
        for n,h in c['files'].items():assert b.sha(root/n)==h
        z=np.load(root/'predictions.npz');assert np.array_equal(z['rows'],np.flatnonzero(fold==f))
        for key in MODES:pred[key][z['rows']]=z[key]
    pred['two_seed_mean']=.5*(pred['original']+pred['fresh_kumo_seed']);metrics={}
    for key,p in pred.items():
        assert np.isfinite(p).all();score=b.scores(y,p)
        assert abs(score['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-14
        score['fold_KS']=[b.scores(y[fold==f],p[fold==f])['KS']for f in range(5)];metrics[key]=score
    evaluated={k:pred[k]for k in MODES}
    evaluated.update({f'control_{k}':ref[f'maturity__{k}']for k in MODES})
    info={}
    for name,p in evaluated.items():
        order=np.argsort(p,kind='stable');s=p[order];info[name]=(order,y[order],np.r_[s[1:]!=s[:-1],True])
    groups=[np.flatnonzero((fold==f)&(y==cl))for f in range(5)for cl in (0,1)]
    rng=np.random.default_rng(20261025140);draws={k:[]for k in MODES}
    for _ in range(2000):
        counts=np.zeros(len(y),dtype=int)
        for g in groups:counts[g]=rng.multinomial(len(g),np.full(len(g),1/len(g)))
        values={n:weighted_ks(t,counts[o],e)for n,(o,t,e)in info.items()}
        for key in MODES:draws[key].append(values[key]-values[f'control_{key}'])
    contrasts={}
    for key in MODES:
        contrasts[key]=dict(KS_delta=metrics[key]['KS']-b.scores(y,evaluated[f'control_{key}'])['KS'],
                            two_contrast_adjusted95pct_interval=np.quantile(draws[key],[.0125,.9875]).tolist(),
                            fold_wins=int(sum(metrics[key]['fold_KS'][f]>b.scores(y[fold==f],evaluated[f'control_{key}'][fold==f])['KS']for f in range(5))))
    passed=(metrics['original']['KS']>.5314098055476294 and contrasts['original']['fold_wins']>=3
            and all(c['KS_delta']>0 and c['two_contrast_adjusted95pct_interval'][0]>0 for c in contrasts.values()))
    np.savez_compressed(D/'OOF.npz',target=y,fold=fold,**pred);np.savez_compressed(D/'BOOTSTRAP.npz',**draws)
    out=dict(status='complete',metrics=metrics,comparisons=contrasts,selected='original'if passed else None,
             promotion_passed=bool(passed),target_054_reached=metrics['original']['KS']>=.54,
             hypothesis_sha256=b.sha(R/'H014.json'),independent_confirmation=False,
             limitation='Default labels historically reused; originalseed primary frozen before metrics, freshseed is replication. Two-seed mean descriptive only.')
    b.dump(out,D/'RESULT.json');print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','run','report']);p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
    if a.action=='build':build()
    elif a.action=='run':run(a.fold)
    else:report()
