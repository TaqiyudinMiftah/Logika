"""Freeze a correction-selected paired-seed architecture recipe before terminal scoring."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from common import R,dump,sha,ks,utc


def load(name,fold):
    root=R/f'runs/{name}/fold{fold}';done=json.loads((root/'RESULT.json').read_text());assert done['status']=='complete'and sha(root/'prediction.npz')==done['prediction_sha256']
    p=np.load(root/'prediction.npz');roles=np.load(Path(json.loads((R/'PATHS.json').read_text())['frozen_root'])/f'fold{fold}/roles.npz')
    assert np.array_equal(p['rows'],np.r_[roles['correction'],roles['validation']]);return p['raw']


def freeze():
    configs=json.loads((R/'CONFIGS.json').read_text());paths=json.loads((R/'PATHS.json').read_text());groups={}
    for name,spec in configs.items():groups.setdefault(spec['design'],{})[str(spec['seed'])]=name
    complete={d:g for d,g in groups.items()if set(g)=={'2026','7731'}and all((R/f'runs/{name}/fold{f}/RESULT.json').exists()for name in g.values()for f in range(5))}
    assert complete,'No complete paired-seed design; preserve incumbent and report incomplete.'
    out=R/'evaluation';out.mkdir(exist_ok=False);recipes=[]
    for fold in range(5):
        root=Path(paths['frozen_root'])/f'fold{fold}';roles=np.load(root/'roles.npz');y=roles['y_correction'];base=np.load(root/'baseline.npz')['raw']
        ranking,choice=train_test_split(np.arange(6000),test_size=.5,stratify=y,random_state=261026)
        predictions={d:np.mean([load(name,fold)for name in g.values()],axis=0)for d,g in complete.items()}
        order=sorted(complete,key=lambda d:(-ks(y[ranking],predictions[d][ranking]),d));best=None;trials=[]
        for count in sorted(set([1,min(2,len(order)),min(4,len(order)),len(order)])):
            avg=np.mean([predictions[d]for d in order[:count]],axis=0)
            for weight in [0,.05,.1,.2,.4,.7,1.]:
                p=(1-weight)*base+weight*avg;score=ks(y[choice],p[choice]);key=(score,-weight,-count);trials.append({'members':count,'weight':weight,'choice_KS':score})
                if best is None or key>best[0]:best=(key,p,order[:count],weight)
        np.savez_compressed(out/f'fold{fold}_primary.npz',raw=best[1],rows=np.r_[roles['correction'],roles['validation']])
        recipes.append({'fold':fold,'designs':best[2],'weight':best[3],'choice_KS':best[0][0],'trials':trials,'outer_labels_loaded':False})
    dump({'status':'frozen','UTC':utc(),'recipes':recipes,'complete_designs':complete,'expected_designs':len(groups),'feature_freeze_sha256':sha(R/'FEATURE_FREEZE.json'),
          'configs_sha256':sha(R/'CONFIGS.json'),'contract_sha256':sha(R/'VALIDATION_CONTRACT.json'),'source_sha256':sha(__file__)},out/'FROZEN.json')


def interval(y,p,base):
    positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
    def prep(values):
        order=np.argsort(values,kind='stable');return order,np.r_[np.flatnonzero(np.diff(values[order])!=0),len(y)-1]
    def weighted(w,prepared):
        order,ends=prepared;increments=np.where(y[order]==1,w[order]/w[positive].sum(),-w[order]/w[negative].sum())
        return float(np.max(abs(np.cumsum(increments)[ends])))
    a,b=prep(p),prep(base);rng=np.random.default_rng(271026);deltas=[]
    for _ in range(2000):
        rows=np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)];weights=np.bincount(rows,minlength=len(y));deltas.append(weighted(weights,a)-weighted(weights,b))
    return {'delta_KS':ks(y,p)-ks(y,base),'percentile95':np.quantile(deltas,[.025,.975]).tolist(),'replicates':2000,'unit':'application,class-stratified','limits':'Conditional adaptively reused development evidence; not independentconfirmation.'}


def evaluate():
    frozen=json.loads((R/'evaluation/FROZEN.json').read_text());assert frozen['source_sha256']==sha(__file__)
    paths=json.loads((R/'PATHS.json').read_text());sys.path.insert(0,paths['old_head_source']);import baseline as b;import feature_head
    old=np.load(paths['old_labels']);y,folds,base=old['target'],old['fold'],old['maturity__original'];variants={};raw_variants={};replay_error=0.
    for fold in range(5):
        root=Path(paths['frozen_root'])/f'fold{fold}';roles=np.load(root/'roles.npz');rows=roles['validation'];manifest=json.loads((root/'MANIFEST.json').read_text());baseline=np.load(root/'baseline.npz')
        table=pd.read_parquet(root/'features.parquet').iloc[np.r_[roles['correction'],rows]].reset_index(drop=True)
        candidates={};pairs={}
        for design,g in frozen['complete_designs'].items():
            for seed,name in g.items():candidates[name]=load(name,fold)
            pairs[design]=np.mean([candidates[name]for name in g.values()],axis=0)
        candidates.update(pairs);candidates['primary']=np.load(R/f'evaluation/fold{fold}_primary.npz')['raw'];candidates['baseline_replay']=baseline['raw']
        recipe=next(v for v in frozen['recipes']if v['fold']==fold)
        for seed in ['2026','7731']:
            member=np.mean([candidates[frozen['complete_designs'][d][seed]]for d in recipe['designs']],axis=0)
            candidates['primary_seed'+seed]=(1-recipe['weight'])*baseline['raw']+recipe['weight']*member
        for name,p in candidates.items():
            head=feature_head.fit(table.iloc[:6000].assign(target=roles['y_correction']),p[:6000],manifest['head_columns'],.003,.5)
            pred=.75*b.corrected_probability(head,table.iloc[6000:],p[6000:])[0]+.25*baseline['spline']
            if name=='baseline_replay':replay_error=max(replay_error,float(np.max(abs(pred-base[rows]))))
            if name not in variants:variants[name]=np.full(70000,np.nan);raw_variants[name]=np.full(70000,np.nan)
            variants[name][rows]=pred;raw_variants[name][rows]=p[6000:]
    assert replay_error<1e-10
    metrics={}
    for name,p in variants.items():
        assert np.isfinite(p).all();metrics[name]={'KS':ks(y,p),'raw_KS':ks(y,raw_variants[name]),'AUC':float(roc_auc_score(y,p)),'delta_KS':ks(y,p)-ks(y,base),
                                                 'fold_delta':[ks(y[folds==f],p[folds==f])-ks(y[folds==f],base[folds==f])for f in range(5)]}
    ci=interval(y,variants['primary'],base);seeds={s:metrics['primary_seed'+s]['delta_KS']for s in ['2026','7731']}
    np.savez_compressed(R/'evaluation/OOF.npz',target=y,fold=folds,baseline=base,**variants)
    result={'status':'evaluated','UTC':utc(),'primary':metrics['primary'],'baseline_KS':ks(y,base),'uncertainty':ci,'fixed_recipe_seed_deltas':seeds,
            'complete_designs':len(frozen['complete_designs']),'expected_designs':frozen['expected_designs'],'all_designs_complete':len(frozen['complete_designs'])==frozen['expected_designs'],
            'target_reached':metrics['primary']['KS']>=.54,'development_gain_supported':metrics['primary']['delta_KS']>0 and ci['percentile95'][0]>0 and sum(v>0 for v in metrics['primary']['fold_delta'])>=3 and all(v>0 for v in seeds.values()),
            'independent_confirmation':False,'promotion_eligible':False,'submission_made':False,'diagnostic_metrics':metrics,'baseline_replay_max_abs':replay_error,
            'OOF_sha256':sha(R/'evaluation/OOF.npz'),'frozen_recipe_sha256':sha(R/'evaluation/FROZEN.json'),'contract_sha256':sha(R/'VALIDATION_CONTRACT.json')}
    dump(result,R/'evaluation/RESULT.json');print(json.dumps({k:result[k]for k in ['status','primary','target_reached','development_gain_supported','fixed_recipe_seed_deltas','complete_designs']},indent=2),flush=True)


if __name__=='__main__':
    if not (R/'evaluation/FROZEN.json').exists():freeze()
    evaluate()
