"""Freeze the primary weight without outer labels, then evaluate the fixed procedure."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(value,path):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def ks(y,p):
    f,t,_=roc_curve(y,p,drop_intermediate=False);return float(np.max(abs(t-f)))
def shifted(base,direction,weight):return base.copy() if weight==0 else np.clip(base+weight*direction,1e-6,1-1e-6)
def prediction(size,group,fold):
    selection=json.loads((R/f'selection/fold{fold}.json').read_text())
    if group!='baseline90' and not selection['groups'][group]:group='baseline90'
    root=R/f'runs/{size}8/{group}/fold{fold}';record=json.loads((root/'RESULT.json').read_text())
    assert record['status']=='complete' and record['fitted_weights_saved'] is False
    assert sha(root/'prediction.npz')==record['prediction_sha256']
    z=np.load(root/'prediction.npz');roles=np.load(R/f'packets/fold{fold}/roles.npz')
    assert np.array_equal(z['rows'],np.r_[roles['correction'],roles['validation']])
    return z['raw']
def directions(fold):
    selection=json.loads((R/f'selection/fold{fold}.json').read_text())
    controls={size:prediction(size,'baseline90',fold) for size in ['medium','large']}
    control=np.mean(list(controls.values()),axis=0)
    augmented={group:np.mean([prediction(size,group,fold) for size in ['medium','large']],axis=0) for group in selection['groups']}
    difference={group:p-control for group,p in augmented.items()}
    primary=np.mean([difference[g] for g in selection['families']],axis=0)
    return primary,difference,controls,augmented
def choose_weight(y,base,direction,choice,plan,fold):
    splitter=StratifiedShuffleSplit(n_splits=5,train_size=.8,random_state=261026+fold)
    subsets=[choice[a] for a,b in splitter.split(np.zeros(len(choice)),y[choice])]
    best=(0.,0.);trials=[]
    for weight in plan['primary']['weights']:
        p=shifted(base,direction,weight);gain=ks(y[choice],p[choice])-ks(y[choice],base[choice])
        gains=[ks(y[a],p[a])-ks(y[a],base[a]) for a in subsets]
        stable=sum(g>0 for g in gains)>=4
        admissible=weight==0 or (gain>0 and stable)
        trials.append({'weight':weight,'choice_delta_KS':gain,'subsample_delta_KS':gains,'admissible':admissible})
        key=(gain,-weight)
        if admissible and key>best:best=key
    return -best[1],trials
def freeze():
    plan=json.loads((R/'PLAN.json').read_text());out=R/'evaluation';out.mkdir(exist_ok=False);recipes=[]
    for fold in range(5):
        roles=np.load(R/f'packets/fold{fold}/roles.npz');y=roles['y_correction'];base=np.load(R/f'packets/fold{fold}/baseline.npz')['raw']
        ranking,choice=train_test_split(np.arange(6000),test_size=.5,stratify=y,random_state=261026)
        direction,_,_,_=directions(fold);weight,trials=choose_weight(y,base,direction,choice,plan,fold)
        np.savez_compressed(out/f'fold{fold}_primary.npz',raw=shifted(base,direction,weight),rows=np.r_[roles['correction'],roles['validation']])
        recipes.append({'fold':fold,'weight':weight,'trials':trials,'outer_labels_loaded':False})
    dump({'status':'frozen','recipes':recipes,'plan_sha256':sha(R/'PLAN.json'),'source_sha256':sha(__file__),
          'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},out/'FROZEN.json')
def bootstrap(y,p,base):
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    def prepare(values):
        ix=np.argsort(values,kind='stable');return ix,np.r_[np.flatnonzero(np.diff(values[ix])!=0),len(y)-1]
    a,b=prepare(p),prepare(base)
    def weighted(w,pre):
        ix,ends=pre;signed=np.where(y[ix]==1,w[ix]/w[pos].sum(),-w[ix]/w[neg].sum())
        return float(np.max(abs(np.cumsum(signed)[ends])))
    rng=np.random.default_rng(261026);d=[]
    for _ in range(2000):
        ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        w=np.bincount(ix,minlength=len(y));d.append(weighted(w,a)-weighted(w,b))
    return {'delta_KS':ks(y,p)-ks(y,base),'percentile_95':np.quantile(d,[.025,.975]).tolist(),'replicates':2000,'seed':261026,
            'limits':'Application-stratified conditional reused-development uncertainty; no independent audit and no second training seed.'}
def evaluate():
    frozen=json.loads((R/'evaluation/FROZEN.json').read_text());assert frozen['source_sha256']==sha(__file__)
    sys.path.insert(0,'/mnt/home/faris/logika-faris-exploration/transfer-20261025/source')
    import baseline as b
    import feature_head
    ref=np.load('/mnt/home/faris/logika-faris-exploration/transfer-20261025/ROBUSTNESS_OOF.npz')
    y,folds,base_oof=ref['target'],ref['fold'],ref['maturity__original'];variants={};max_baseline_error=0.
    for fold in range(5):
        roles=np.load(R/f'packets/fold{fold}/roles.npz');rows=roles['validation'];yc=roles['y_correction']
        import pandas as pd
        frame=pd.read_parquet(R/f'packets/fold{fold}/current.parquet').iloc[np.r_[roles['correction'],rows]].reset_index(drop=True)
        m=json.loads((R/f'packets/fold{fold}/manifest.json').read_text());old=np.load(R/f'packets/fold{fold}/baseline.npz');raw=old['raw']
        direction,difference,controls,augmented=directions(fold)
        candidates={'primary':np.load(R/f'evaluation/fold{fold}_primary.npz')['raw'],'baseline_replay':raw}
        candidates.update({'control_'+size:p for size,p in controls.items()})
        for group,p in augmented.items():
            candidates['paired_'+group]=shifted(raw,difference[group],.2)
            candidates['augmented_'+group]=p
        for name,p in candidates.items():
            head=feature_head.fit(frame.iloc[:6000].assign(target=yc),p[:6000],m['head_columns'],.003,.5)
            corrected=.75*b.corrected_probability(head,frame.iloc[6000:],p[6000:])[0]+.25*old['spline']
            if name=='baseline_replay':max_baseline_error=max(max_baseline_error,float(np.max(abs(corrected-base_oof[rows]))))
            if name not in variants:variants[name]=np.full(len(y),np.nan)
            variants[name][rows]=corrected
    assert max_baseline_error<1e-10,max_baseline_error
    metrics={}
    for name,p in variants.items():
        assert np.isfinite(p).all()
        metrics[name]={'KS':ks(y,p),'AUC':float(roc_auc_score(y,p)),'delta_KS':ks(y,p)-ks(y,base_oof),
                       'fold_delta':[ks(y[folds==f],p[folds==f])-ks(y[folds==f],base_oof[folds==f]) for f in range(5)]}
    uncertainty=bootstrap(y,variants['primary'],base_oof)
    np.savez_compressed(R/'evaluation/OOF.npz',target=y,fold=folds,baseline=base_oof,**variants)
    result={'status':'evaluated','primary':metrics['primary'],'baseline_KS':ks(y,base_oof),'diagnostic_metrics':metrics,
            'uncertainty':uncertainty,'target_reached':metrics['primary']['KS']>=.54,'independent_confirmation':False,
            'seed_replication_required_before_promotion':metrics['primary']['delta_KS']>0,'promotion_eligible':False,
            'baseline_replay_max_abs':max_baseline_error,'frozen_recipe_sha256':sha(R/'evaluation/FROZEN.json'),
            'OOF_sha256':sha(R/'evaluation/OOF.npz'),'submission_made':False,
            'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    dump(result,R/'evaluation/RESULT.json');print(json.dumps({k:result[k] for k in ['status','primary','baseline_KS','target_reached','promotion_eligible']},indent=2))
def main():
    if not (R/'evaluation/FROZEN.json').exists():freeze()
    evaluate()
if __name__=='__main__':main()
