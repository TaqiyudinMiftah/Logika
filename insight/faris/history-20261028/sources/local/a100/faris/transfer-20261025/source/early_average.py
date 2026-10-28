"""H009: fixed early seed averaging, with the objective as a matched ablation."""
import argparse,time
import numpy as np
import joblib
from scipy.stats import ks_2samp
import baseline as b
import feature_head
from direct_ks import fit as direct_fit
from robustness import R,inputs,stages,read,check

def run(f):
    check();plan=read(R/'H009.json')
    for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
    root=R/f'early_average/fold{f}';root.mkdir(parents=True,exist_ok=False)
    original,ix,q,context,yc,columns,oldraw,frame,mature,models=inputs(f)
    q,table=stages(q,frame,mature,'periodicity')
    fresh=np.load(R/f'robustness/fold{f}/raw_seed_repeat.npz')['prediction'];raw=.5*(oldraw+fresh)
    np.savez_compressed(root/'raw_mean.npz',rows=np.r_[ix['correction'],ix['validation']],application_id=q.application_id.to_numpy(str),prediction=raw)
    spline=b.predict_spline(models['periodicity'][1],table.iloc[ix['validation']]);training=q.iloc[:6000].copy();training['target']=yc
    started=time.time();anchor=feature_head.fit(training,raw[:6000],models['periodicity'][0]['columns'],.003,.5)
    for mode,head in [('logloss',anchor),('direct_KS',direct_fit(anchor,training,raw[:6000]))]:
        file=root/f'{mode}.joblib';joblib.dump(head,file,compress=3)
        pred=.75*b.corrected_probability(head,q.iloc[6000:],raw[6000:])[0]+.25*spline
        replay=.75*b.corrected_probability(joblib.load(file),q.iloc[6000:],raw[6000:])[0]+.25*spline
        assert np.array_equal(pred,replay)
        np.savez_compressed(root/f'{mode}.npz',rows=ix['validation'],application_id=q.application_id.iloc[6000:].to_numpy(str),prediction=pred)
    b.dump(dict(status='complete',fold=f,seconds=time.time()-started,training_rows=6000,validation_rows=14000,raw_seed_weights=[.5,.5],head_strength=.5,head_penalty=.003,source_sha256=plan['source_sha256'],files={p.name:b.sha(p)for p in root.iterdir()if p.is_file()}),root/'complete.json')

def report():
    z=np.load(R/'ROBUSTNESS_OOF.npz');y,fold=z['target'],z['fold'];predictions={};metrics={};control=z['periodicity__two_seed_mean']
    for mode in ['logloss','direct_KS']:
        p=np.full(len(y),np.nan)
        for f in range(5):
            root=R/f'early_average/fold{f}';record=read(root/'complete.json');assert record['status']=='complete'
            for n,h in record['files'].items():assert b.sha(root/n)==h
            a=np.load(root/f'{mode}.npz');assert np.array_equal(a['rows'],np.flatnonzero(fold==f));p[a['rows']]=a['prediction']
        assert np.isfinite(p).all();predictions[mode]=p;score=b.scores(y,p)
        assert abs(score['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-14
        diffs=[b.scores(y[fold==f],p[fold==f])['KS']-b.scores(y[fold==f],control[fold==f])['KS']for f in range(5)]
        metrics[mode]=dict(score=score,KS_delta_vs_A=score['KS']-b.scores(y,control)['KS'],fold_KS_delta=diffs,fold_wins=int(sum(d>0 for d in diffs)))
    np.savez_compressed(R/'EARLY_AVERAGE_OOF.npz',target=y,fold=fold,**predictions)
    def ks(yy,p):
        order=np.argsort(p,kind='stable');t=yy[order];s=p[order];ends=np.r_[s[1:]!=s[:-1],True];pos=np.cumsum(t);neg=np.arange(1,len(t)+1)-pos
        return float(np.max(np.abs((neg/(len(t)-t.sum())-pos/t.sum())[ends])))
    groups=[np.flatnonzero((fold==f)&(y==cl))for f in range(5)for cl in(0,1)]
    rng=np.random.default_rng(202610259);draws={k:[]for k in predictions}
    for _ in range(1000):
        rows=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);base=ks(y[rows],control[rows])
        for mode,p in predictions.items():draws[mode].append(ks(y[rows],p[rows])-base)
    for mode in predictions:
        m=metrics[mode];m['conditional95percent_interval']=np.quantile(draws[mode],[.025,.975]).tolist();m['two_contrast_adjusted95percent_interval']=np.quantile(draws[mode],[.0125,.9875]).tolist()
        m['promotion_passed']=bool(m['KS_delta_vs_A']>0 and m['fold_wins']>=3 and m['two_contrast_adjusted95percent_interval'][0]>0)
    eligible=[k for k in metrics if metrics[k]['promotion_passed']];selected=max(eligible,key=lambda k:metrics[k]['score']['KS'])if eligible else None
    b.dump(dict(status='complete',metrics=metrics,selected=selected,target_054_reached=any(metrics[k]['score']['KS']>=.54 for k in metrics),independent_confirmation=False,uncertainty_limitation='Conditional on fixed predictions and reused development labels; excludes adaptive search and full refitting.'),R/'EARLY_AVERAGE_RESULT.json')
    np.savez_compressed(R/'EARLY_AVERAGE_BOOTSTRAP.npz',**draws)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','report']);p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args();run(a.fold)if a.action=='run'else report()
