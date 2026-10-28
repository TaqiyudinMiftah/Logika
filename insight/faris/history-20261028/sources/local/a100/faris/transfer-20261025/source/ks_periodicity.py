"""H008: one compatible crossover of the fixed objective with the periodicity recipe."""
import argparse,time
import numpy as np
import joblib
from scipy.stats import ks_2samp
import baseline as b
from direct_ks import fit,TEMPERATURE,TRUST,RIDGE
from robustness import R,inputs,stages,read,check

def run(f):
    check();plan=read(R/'H008.json')
    for name,digest in plan['source_sha256'].items():assert b.sha(R/'source'/name)==digest
    root=R/f'ks_periodicity/fold{f}';root.mkdir(parents=True,exist_ok=False)
    original,ix,q,context,yc,columns,oldraw,frame,mature,models=inputs(f)
    q,table=stages(q,frame,mature,'periodicity')
    spline=b.predict_spline(models['periodicity'][1],table.iloc[ix['validation']])
    training=q.iloc[:6000].copy();training['target']=yc
    control=np.load(R/'ROBUSTNESS_OOF.npz')
    started=time.time()
    for mode in ['original','fresh_kumo_seed']:
        if mode=='original':head=models['periodicity'][0];raw=oldraw
        else:
            head=joblib.load(R/f'robustness/fold{f}/periodicity__fresh_kumo_seed.joblib')
            raw=np.load(R/f'robustness/fold{f}/raw_seed_repeat.npz')['prediction']
        reference=.75*b.corrected_probability(head,q.iloc[6000:],raw[6000:])[0]+.25*spline
        expected=control['periodicity__'+mode][ix['validation']]
        assert np.max(np.abs(reference-expected))<2e-14
        new=fit(head,training,raw[:6000]);file=root/f'{mode}.joblib';joblib.dump(new,file,compress=3)
        pred=.75*b.corrected_probability(new,q.iloc[6000:],raw[6000:])[0]+.25*spline
        replay=.75*b.corrected_probability(joblib.load(file),q.iloc[6000:],raw[6000:])[0]+.25*spline
        assert np.array_equal(pred,replay)
        np.savez_compressed(root/f'{mode}.npz',rows=ix['validation'],application_id=q.application_id.iloc[6000:].to_numpy(str),prediction=pred)
    b.dump(dict(status='complete',fold=f,seconds=time.time()-started,head_rows=6000,validation_rows=14000,control_replay_passed=True,source_sha256=plan['source_sha256'],files={p.name:b.sha(p)for p in root.iterdir()if p.is_file()}),root/'complete.json')

def report():
    z=np.load(R/'ROBUSTNESS_OOF.npz');y,fold=z['target'],z['fold'];predictions={};metrics={};effects={}
    for mode in ['original','fresh_kumo_seed']:
        p=np.full(len(y),np.nan)
        for f in range(5):
            root=R/f'ks_periodicity/fold{f}';record=read(root/'complete.json');assert record['status']=='complete'
            for name,digest in record['files'].items():assert b.sha(root/name)==digest
            a=np.load(root/f'{mode}.npz');assert np.array_equal(a['rows'],np.flatnonzero(fold==f));p[a['rows']]=a['prediction']
        assert np.isfinite(p).all();predictions[mode]=p
    predictions['two_seed_mean']=.5*(predictions['original']+predictions['fresh_kumo_seed'])
    for mode,p in predictions.items():
        score=b.scores(y,p);assert abs(score['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-14
        ref=z['periodicity__'+mode];diff=[b.scores(y[fold==f],p[fold==f])['KS']-b.scores(y[fold==f],ref[fold==f])['KS']for f in range(5)]
        metrics[mode]=score;effects[mode]=dict(KS_delta=score['KS']-b.scores(y,ref)['KS'],fold_KS_delta=diff,fold_wins=int(sum(d>0 for d in diff)))
    accepted=all(effects[m]['KS_delta']>0 for m in effects)and effects['two_seed_mean']['fold_wins']>=3
    b.dump(dict(status='complete',metrics=metrics,effects=effects,point_rule_passed=accepted,submission_requires_paired_lower95_above_zero=True,target_054_reached=metrics['two_seed_mean']['KS']>=.54,independent_confirmation=False),R/'KS_PERIODICITY_RESULT.json')
    np.savez_compressed(R/'KS_PERIODICITY_OOF.npz',target=y,fold=fold,**predictions)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','report']);p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
    run(a.fold)if a.action=='run'else report()
