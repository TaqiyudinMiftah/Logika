"""Choose only on correction rows; outer labels enter after recipe freeze."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from common import R,dump,sha,ks,utc
def load(config,fold):
    root=R/f'runs/{config}/fold{fold}';record=json.loads((root/'RESULT.json').read_text());assert record['status']=='complete'
    assert sha(root/'prediction.npz')==record['prediction_sha256'];z=np.load(root/'prediction.npz');roles=np.load(R/f'frozen/fold{fold}/roles.npz')
    assert np.array_equal(z['rows'],np.r_[roles['correction'],roles['validation']]);return z['raw']
def freeze():
    configs=json.loads((R/'CONFIGS.json').read_text());complete=[c for c in configs if all((R/f'runs/{c}/fold{f}/RESULT.json').exists() for f in range(5))]
    assert len(complete)>=2,'Insufficient complete configurations'
    out=R/'evaluation';out.mkdir(exist_ok=False);recipes=[]
    for fold in range(5):
        roles=np.load(R/f'frozen/fold{fold}/roles.npz');y=roles['y_correction'];base=np.load(R/f'frozen/fold{fold}/baseline.npz')['raw']
        ranking,choice=train_test_split(np.arange(6000),test_size=.5,stratify=y,random_state=261026)
        predictions={c:load(c,fold) for c in complete};order=sorted(complete,key=lambda c:(-ks(y[ranking],predictions[c][ranking]),c))
        best=None;trials=[]
        for count in sorted(set([1,min(4,len(order)),min(8,len(order)),len(order)])):
            avg=np.mean([predictions[c] for c in order[:count]],axis=0)
            for weight in [0,.1,.2,.4,.7,1.]:
                pred=(1-weight)*base+weight*avg;score=ks(y[choice],pred[choice]);key=(score,-weight,-count)
                trials.append({'members':count,'weight':weight,'choice_KS':score})
                if best is None or key>best[0]:best=(key,pred,order[:count],weight)
        np.savez_compressed(out/f'fold{fold}_primary.npz',raw=best[1],rows=np.r_[roles['correction'],roles['validation']])
        recipes.append({'fold':fold,'members':best[2],'weight':best[3],'choice_KS':best[0][0],'trials':trials,'outer_labels_loaded':False})
    dump({'status':'frozen','UTC':utc(),'recipes':recipes,'complete_configs':complete,'expected_configs':len(configs),
          'feature_freeze_sha256':sha(R/'FEATURE_FREEZE.json'),'configs_sha256':sha(R/'CONFIGS.json'),'source_sha256':sha(__file__)},out/'FROZEN.json')
def interval(y,p,base):
    positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
    def prep(v):
        order=np.argsort(v,kind='stable');return order,np.r_[np.flatnonzero(np.diff(v[order])!=0),len(y)-1]
    def weighted(w,prepared):
        order,ends=prepared;s=np.where(y[order]==1,w[order]/w[positive].sum(),-w[order]/w[negative].sum())
        return float(np.max(abs(np.cumsum(s)[ends])))
    a,b=prep(p),prep(base);rng=np.random.default_rng(261026);values=[]
    for _ in range(2000):
        rows=np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)]
        w=np.bincount(rows,minlength=len(y));values.append(weighted(w,a)-weighted(w,b))
    return {'delta_KS':ks(y,p)-ks(y,base),'percentile95':np.quantile(values,[.025,.975]).tolist(),'replicates':2000,
            'unit':'application, resampled separately within classes','limits':'Conditional adaptively reused development evidence. Independent confirmation unavailable.'}
def evaluate():
    frozen=json.loads((R/'evaluation/FROZEN.json').read_text());assert frozen['source_sha256']==sha(__file__)
    sys.path.insert(0,'/mnt/home/faris/logika-faris-exploration/transfer-20261025/source')
    import baseline as b
    import feature_head
    old=np.load('/mnt/home/faris/logika-faris-exploration/transfer-20261025/ROBUSTNESS_OOF.npz');y,folds,baseline=old['target'],old['fold'],old['maturity__original']
    variants={};raw_variants={};replay_error=0.
    for fold in range(5):
        root=R/f'frozen/fold{fold}';roles=np.load(root/'roles.npz');rows=roles['validation'];yc=roles['y_correction']
        table=pd.read_parquet(root/'features.parquet').iloc[np.r_[roles['correction'],rows]].reset_index(drop=True)
        m=json.loads((root/'MANIFEST.json').read_text());base=np.load(root/'baseline.npz')
        candidates={c:load(c,fold) for c in frozen['complete_configs']}
        candidates.update(primary=np.load(R/f'evaluation/fold{fold}_primary.npz')['raw'],baseline_replay=base['raw'])
        for name,p in candidates.items():
            head=feature_head.fit(table.iloc[:6000].assign(target=yc),p[:6000],m['head_columns'],.003,.5)
            pred=.75*b.corrected_probability(head,table.iloc[6000:],p[6000:])[0]+.25*base['spline']
            if name=='baseline_replay':replay_error=max(replay_error,float(np.max(abs(pred-baseline[rows]))))
            if name not in variants:variants[name]=np.full(70000,np.nan);raw_variants[name]=np.full(70000,np.nan)
            variants[name][rows]=pred;raw_variants[name][rows]=p[6000:]
    assert replay_error<1e-10,replay_error
    metrics={}
    for name,p in variants.items():
        assert np.isfinite(p).all()
        metrics[name]={'KS':ks(y,p),'raw_KS':ks(y,raw_variants[name]),'AUC':float(roc_auc_score(y,p)),
                       'delta_KS':ks(y,p)-ks(y,baseline),'fold_delta':[ks(y[folds==f],p[folds==f])-ks(y[folds==f],baseline[folds==f]) for f in range(5)]}
    uncertainty=interval(y,variants['primary'],baseline)
    np.savez_compressed(R/'evaluation/OOF.npz',target=y,fold=folds,baseline=baseline,**variants)
    fullcoverage=len(frozen['complete_configs'])==frozen['expected_configs']
    result={'status':'evaluated','UTC':utc(),'primary':metrics['primary'],'baseline_KS':ks(y,baseline),
            'target_reached':metrics['primary']['KS']>=.54,'uncertainty':uncertainty,'diagnostic_metrics':metrics,
            'all_configs_complete':fullcoverage,'complete_configs':len(frozen['complete_configs']),
            'development_gain_supported':metrics['primary']['delta_KS']>0 and uncertainty['percentile95'][0]>0 and sum(v>0 for v in metrics['primary']['fold_delta'])>=3,
            'independent_confirmation':False,'promotion_eligible':False,'submission_made':False,'baseline_replay_max_abs':replay_error,
            'OOF_sha256':sha(R/'evaluation/OOF.npz'),'frozen_recipe_sha256':sha(R/'evaluation/FROZEN.json')}
    dump(result,R/'evaluation/RESULT.json');print(json.dumps({k:result[k] for k in ['status','primary','target_reached','development_gain_supported','all_configs_complete']},indent=2),flush=True)
def main():
    if not (R/'evaluation/FROZEN.json').exists():freeze()
    evaluate()
if __name__=='__main__':main()
