"""Inner-only subspace selection and outer evaluation of the frozen procedure."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
R=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(y,p):
    f,t,_=roc_curve(y,p,drop_intermediate=False)
    return float(np.max(abs(t-f)))
def dump(v,p):p.write_text(json.dumps(v,indent=2,default=str,allow_nan=False)+'\n')
def complete_packs(size='small',estimators=2):
    catalog=json.loads((R/'CATALOG.json').read_text())
    complete=[]
    for p in catalog['packs']:
        if all((R/f'runs/{size}{estimators}/{p["id"]}/fold{f}/RESULT.json').exists() for f in range(5)):
            complete.append(p)
    return complete
def load(pack,fold,size='small',estimators=2):
    root=R/f'runs/{size}{estimators}/{pack}/fold{fold}'
    done=json.loads((root/'RESULT.json').read_text())
    assert done['status']=='complete' and sha(root/'prediction.npz')==done['prediction_sha256']
    z=np.load(root/'prediction.npz');roles=np.load(R/f'packets/fold{fold}/roles.npz')
    assert np.array_equal(z['rows'],np.r_[roles['correction'],roles['validation']])
    return z['raw']
def select(allow_partial=False):
    plan=json.loads((R/'PLAN.json').read_text());packs=complete_packs()
    expected=json.loads((R/'CATALOG.json').read_text())['pack_count']
    if not allow_partial:assert len(packs)==expected,(len(packs),expected)
    assert len(packs)>=3
    out=R/'selection';out.mkdir(exist_ok=False)
    tasks=[];selection=[]
    for fold in range(5):
        roles=np.load(R/f'packets/fold{fold}/roles.npz');y=roles['y_correction']
        ranking,choice=train_test_split(np.arange(6000),test_size=.5,stratify=y,random_state=261026)
        base=np.load(R/f'packets/fold{fold}/baseline.npz')['raw']
        predictions={p['id']:load(p['id'],fold) for p in packs}
        order=sorted(predictions,key=lambda k:(-score(y[ranking],predictions[k][ranking]),k))
        counts=list(dict.fromkeys(min(len(order),n) if isinstance(n,int) else len(order) for n in plan['ensemble_rule']['counts']))
        best=None;trials=[]
        for count in counts:
            average=np.mean([predictions[k] for k in order[:count]],axis=0)
            for weight in plan['ensemble_rule']['weights']:
                mixed=(1-weight)*base+weight*average
                ks=score(y[choice],mixed[choice]);candidate=(ks,-weight,-count)
                trials.append({'members':count,'weight':weight,'inner_choice_KS':ks})
                if best is None or candidate>best[0]:best=(candidate,count,weight,mixed)
        _,count,weight,mixed=best
        np.savez_compressed(out/f'fold{fold}_screened.npz',raw=mixed,rows=np.r_[roles['correction'],roles['validation']])
        shortlist=order[:plan['confirmation']['shortlist_per_fold']]
        for size in plan['confirmation']['sizes']:
            for k in shortlist:tasks.append({'size':size,'estimators':8,'fold':fold,'pack':k})
        # Matched 90-column controls at the same checkpoint/ensemble count.
        for size in plan['confirmation']['sizes']:
            if 'baseline90' not in shortlist:tasks.append({'size':size,'estimators':8,'fold':fold,'pack':'baseline90'})
        selection.append({'fold':fold,'ranked_packs':order,'selected_members':order[:count],
                          'selected_weight':weight,'choice_KS':best[0][0],'trials':trials,
                          'confirmation_shortlist':shortlist,'outer_labels_loaded':False})
    dump({'status':'frozen','complete_screened_packs':len(packs),'expected_packs':expected,'all_features_screened':len(packs)==expected,
          'selection':selection,'tasks':tasks,'plan_sha256':sha(R/'PLAN.json'),
          'source_sha256':sha(__file__),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},out/'SELECTION.json')
    print(json.dumps({'complete_packs':len(packs),'confirmation_tasks':len(tasks),'status':'selection_frozen'}))

def report(stage):
    # This module is invoked on node2001, where the original evaluation labels live.
    sys.path.insert(0,'/mnt/home/faris/logika-faris-exploration/transfer-20261025/source')
    import baseline as b
    import feature_head
    ref=np.load('/mnt/home/faris/logika-faris-exploration/transfer-20261025/ROBUSTNESS_OOF.npz')
    y,folds=ref['target'],ref['fold'];base=ref['maturity__original']
    plan=json.loads((R/'PLAN.json').read_text());selection=json.loads((R/'selection/SELECTION.json').read_text())
    variants={};recipes=[]
    for fold in range(5):
        root=R/f'packets/fold{fold}';roles=np.load(root/'roles.npz');rows=roles['validation'];yc=roles['y_correction']
        q=__import__('pandas').read_parquet(root/'current.parquet').iloc[np.r_[roles['correction'],rows]].reset_index(drop=True)
        manifest=json.loads((root/'manifest.json').read_text());sp=np.load(root/'baseline.npz')['spline']
        candidates={'screened':np.load(R/f'selection/fold{fold}_screened.npz')['raw']}
        if stage=='confirmation':
            pools={}
            for task in selection['tasks']:
                if task['fold']!=fold:continue
                key=task['size']+'__'+task['pack'];pools[key]=load(task['pack'],fold,task['size'],8)
            ranking,choice=train_test_split(np.arange(6000),test_size=.5,stratify=yc,random_state=261026)
            base_raw=np.load(root/'baseline.npz')['raw']
            order=sorted(pools,key=lambda k:(-score(yc[ranking],pools[k][ranking]),k))
            best=None
            for count in sorted(set([1,min(4,len(order)),len(order)])):
                avg=np.mean([pools[k] for k in order[:count]],axis=0)
                for weight in plan['ensemble_rule']['weights']:
                    pred=(1-weight)*base_raw+weight*avg;key=(score(yc[choice],pred[choice]),-weight,-count)
                    if best is None or key>best[0]:best=(key,pred,order[:count],weight)
            candidates['confirmed']=best[1]
            recipes.append({'fold':fold,'confirmation_members':best[2],'weight':best[3],'inner_choice_KS':best[0][0]})
        for name,pred in candidates.items():
            head=feature_head.fit(q.iloc[:6000].assign(target=yc),pred[:6000],manifest['head_columns'],.003,.5)
            corrected=.75*b.corrected_probability(head,q.iloc[6000:],pred[6000:])[0]+.25*sp
            if name not in variants:variants[name]=np.full(len(y),np.nan)
            variants[name][rows]=corrected
    metrics={}
    for name,pred in variants.items():
        assert np.isfinite(pred).all()
        metrics[name]={**b.scores(y,pred),'delta_KS':score(y,pred)-score(y,base),
                       'fold_delta':[score(y[folds==f],pred[folds==f])-score(y[folds==f],base[folds==f]) for f in range(5)]}
    out=R/'reports';out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'{stage}_OOF.npz',target=y,fold=folds,baseline=base,**variants)
    dump({'stage':stage,'metrics':metrics,'recipes':recipes,'baseline_KS':score(y,base),
          'target_reached':any(v['KS']>=.54 for v in metrics.values()),'independent_confirmation':False,
          'selection_sha256':sha(R/'selection/SELECTION.json'),'source_sha256':sha(__file__)},out/f'{stage}_RESULT.json')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','screening','confirmation']);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    if a.action=='select':select(a.allow_partial)
    else:report(a.action)
