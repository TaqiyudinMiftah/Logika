"""Small genuine nested refit benchmark anchored to the preserved champion.

Historical data reuse is disclosed. Outer labels never select the addition or
weight inside that outer fold. No model choice is trained on global OOF data.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import concurrent.futures as cf
import copy,contextlib,fcntl,hashlib,json,sys,time,traceback
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
sys.path.insert(0,str(SOURCE))
from features import ExploreBank
from adapter import fold_fit
from metaexp.pipeline import fit_predict,predict
from utils import read,write,sha,ks,ident
_BANK=None


def bank():
    global _BANK
    if _BANK is None:
        p=read(ROOT/'benchmark_plan.json')
        _BANK=ExploreBank(Path(p['bank']).parent,p['extras'])
    return _BANK


class Context:
    def __init__(self,folder,plan):
        self.root=folder;self.inputs={'bank':plan['bank'],'extras':plan['extras']}
        self.family='cpu';self.spec={'resources':{'cpus':1,'gpus':0}};self.models=[]
    def check_stop(self):
        assert time.time()<read(ROOT/'clock.json')['deadline'],'Absolute build safety deadline expired'
        assert not (ROOT/'STOP').exists(),'Operator stop requested'
    def path(self,name):
        p=self.root/name;assert p.resolve().is_relative_to(self.root.resolve());p.parent.mkdir(parents=True,exist_ok=True);return p
    def model(self,name):
        assert self.path(name).exists();self.models.append(name)
    def save_json(self,name,value):write(self.path(name),value)


def fit(task):
    plan=read(ROOT/'benchmark_plan.json');folder=ROOT/'fits'/task['key']
    assert len(task['fitrows'])==len(task['y'])
    assert not np.intersect1d(task['fitrows'],task['queryrows']).size
    request={k:v for k,v in task.items() if k not in ['fitrows','queryrows','y']}
    request.update(fit_rows=ident(task['fitrows'].tolist()),query_rows=ident(task['queryrows'].tolist()),labels=ident(task['y'].tolist()),plan_sha=sha(ROOT/'benchmark_plan.json'))
    if (folder/'complete.json').exists():
        assert read(folder/'request.json')==request
        for name,h in read(folder/'complete.json')['files'].items():assert sha(folder/name)==h
        return str(folder)
    assert not folder.exists(),('Retain interrupted attempt for investigation',str(folder))
    folder.mkdir(parents=True);write(folder/'request.json',request);ctx=Context(folder,plan)
    ctx.check_stop();b=bank();start=time.time();component=task['component']
    try:
        with (folder/'training.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            if component['kind']=='resolved':
                from metaexp.pipeline import Preprocessor
                kind=component['model'];params=copy.deepcopy(component['parameters'])
                if kind=='cat':
                    from catboost import CatBoostClassifier as Model
                    params.update(random_seed=task['seed'],thread_count=1,task_type='CPU');params.pop('devices',None);params.pop('gpu_ram_part',None)
                elif kind=='lgb':
                    from lightgbm import LGBMClassifier as Model
                    params.update(random_state=task['seed'],n_jobs=1)
                elif kind=='xgb':
                    from xgboost import XGBClassifier as Model
                    params.update(random_state=task['seed'],n_jobs=1,device='cpu')
                else:
                    from sklearn.linear_model import LogisticRegression as Model
                    params.update(random_state=task['seed'])
                pre=Preprocessor(b,copy.deepcopy(component['representation']),kind,task['seed']);x=pre.fit_transform(task['fitrows'],task['y']);model=Model(**params);model.fit(x,task['y']);weight=component['weight'];values=predict(model,pre.transform(task['queryrows']),weight)
                pre.bank=None;joblib.dump({'preprocessing':pre,'model':model,'weight':weight},folder/'pipeline.joblib',compress=1)
                write(folder/'audit.json',{'preprocessing':pre.audit,'parameters':params})
            elif component['kind']=='historical':
                values,pre,model,weight,cdf,warnings=fit_predict(b,copy.deepcopy(component['recipe']),task['fitrows'],task['y'],task['queryrows'],'cpu',task['seed'])
                pre.bank=None;joblib.dump({'preprocessing':pre,'model':model,'weight':weight},folder/'pipeline.joblib',compress=1)
                write(folder/'audit.json',{'preprocessing':pre.audit,'warnings':warnings})
            else:
                p=copy.deepcopy(component['configuration']);rep=copy.deepcopy(component['representation']);rep['config']['weight']=p['weight']
                values=fold_fit(ctx,b,p,rep,task['fitrows'],task['y'],task['queryrows'],'model',task['seed'])
            assert values.shape==(len(task['queryrows']),) and np.isfinite(values).all()
            np.savez_compressed(folder/'prediction.npz',rows=task['queryrows'],prediction=values)
        write(folder/'complete.json',{'seconds':time.time()-start,'files':{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='complete.json'}})
        return str(folder)
    except BaseException:
        write(folder/'failure.json',{'error':traceback.format_exc()});raise


def choose(labels,champion,candidates,folds):
    """Frozen, low-dimensional inner-only selection with a champion fallback."""
    baseline=ks(labels,champion);options=[]
    for key,p in candidates.items():
        for w in [.05,.10]:
            q=(1-w)*champion+w*p
            ds=np.array([ks(labels[folds==f],q[folds==f])-ks(labels[folds==f],champion[folds==f]) for f in range(3)])
            delta=ks(labels,q)-baseline
            utility=float(ds.mean()-.5*ds.std())
            eligible=sum(ds>0)>=2 and ds.min()>=-.002 and utility>0 and delta>=-.003
            options.append({'component':key,'weight':w,'delta':delta,'fold_delta':ds.tolist(),'utility':utility,'eligible':bool(eligible)})
    valid=[r for r in options if r['eligible']]
    valid.sort(key=lambda r:(r['utility'],r['delta'],-r['weight'],r['component']),reverse=True)
    selected=valid[0] if valid else {'component':None,'weight':0.,'utility':0.}
    return {'selected':selected,'options':options,'uses_only_inner_labels':True}


def tasks_for(prefix,components,rows,labels,query,seed):
    return [{'key':prefix+'_'+c['id'],'component':c,'fitrows':rows,'y':labels,'queryrows':query,'seed':seed} for c in components]


def run_tasks(pool,tasks):
    futures={pool.submit(fit,t):t for t in tasks};result={}
    for f in cf.as_completed(futures):
        t=futures[f];result[t['component']['id']]=f.result()
        write(ROOT/'status.json',{'phase':'nested_refits','current_group_done':len(result),'current_group_total':len(tasks),'last':t['key'],'time':time.time()})
        print(json.dumps({'completed_fit':t['key']}),flush=True)
    return result


def load_predictions(folder,rows):
    z=np.load(Path(folder)/'prediction.npz');assert np.array_equal(z['rows'],rows);return z['prediction']


def trimmed(arrays,ids):return np.sort(np.column_stack([arrays[c] for c in ids]),axis=1)[:,1:-1].mean(1)


def main():
    lock=(ROOT/'benchmark.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ROOT/'benchmark_plan.json')
    for name,h in plan['source_manifest'].items():assert sha(SOURCE/name)==h,('changed source',name)
    for name,h in plan['code_sha256'].items():assert sha(ROOT/name)==h,('changed benchmark code',name)
    for key,h in plan['input_sha256'].items():
        path=Path(plan[key]) if key in ['train','test','sample'] else (Path(plan['bank'])/'manifest.json' if key=='bank_manifest' else Path(plan['extras'])/'manifest.json')
        assert sha(path)==h,('changed input',key)
    for directory in [Path(plan['bank']),Path(plan['extras'])]:
        for name,h in read(directory/'manifest.json')['files'].items():assert sha(directory/name)==h,('changed feature file',name)
    if not (ROOT/'clock.json').exists():write(ROOT/'clock.json',{'start':time.time(),'deadline':time.time()+7200})
    b=bank();rows=np.flatnonzero(b.metadata.split.to_numpy()=='train')
    train=pd.read_csv(plan['train']).set_index('application_id');y=train.target.reindex(b.metadata.application_id.iloc[rows]).to_numpy(dtype=np.int64)
    assert len(rows)==70000 and np.isfinite(y).all()
    components=plan['champion_components']+plan['candidate_components'];champids=[c['id'] for c in plan['champion_components']];candids=[c['id'] for c in plan['candidate_components']]
    outer=list(StratifiedKFold(3,shuffle=True,random_state=20261019).split(rows,y))
    write(ROOT/'evidence/outer_split.json',{'rows':rows.tolist(),
        'outer_validation_rows':[rows[vi].tolist() for _,vi in outer],
        'historical_reuse_disclosed':True,'outer_seed':20261019})
    output=np.full(len(rows),np.nan);champout=np.full(len(rows),np.nan);candidateout={key:np.full(len(rows),np.nan) for key in candids};folds=np.empty(len(rows),int);selections=[]
    import multiprocessing
    with cf.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f,(ti,vi) in enumerate(outer):
            folds[vi]=f
            inner=list(StratifiedKFold(3,shuffle=True,random_state=20261019+117*(f+1)).split(ti,y[ti]))
            innermatrix={c['id']:np.full(len(ti),np.nan) for c in components};innerfold=np.empty(len(ti),int)
            # Every inner fit excludes every outer-validation applicant.
            for g,(a,v) in enumerate(inner):
                innerfold[v]=g
                tasks=tasks_for(f'outer{f}_inner{g}',components,rows[ti[a]],y[ti[a]],rows[ti[v]],2026+f*10+g)
                completed=run_tasks(pool,tasks)
                for key in innermatrix:innermatrix[key][v]=load_predictions(completed[key],rows[ti[v]])
            assert all(np.isfinite(v).all() for v in innermatrix.values())
            champion=trimmed(innermatrix,champids)
            decision=choose(y[ti],champion,{k:innermatrix[k] for k in candids},innerfold)
            write(ROOT/f'evidence/outer{f}_selection.json',decision)
            selected=decision['selected'];need=plan['champion_components']+([c for c in plan['candidate_components'] if c['id']==selected['component']])
            completed=run_tasks(pool,tasks_for(f'outer{f}_refit',need,rows[ti],y[ti],rows[vi],2026+f))
            pred={key:load_predictions(folder,rows[vi]) for key,folder in completed.items()}
            anchor=trimmed(pred,champids);champout[vi]=anchor
            output[vi]=anchor if not selected['component'] else (1-selected['weight'])*anchor+selected['weight']*pred[selected['component']]
            selections.append({'fold':f,**selected})
            np.savez_compressed(ROOT/f'evidence/outer{f}_predictions.npz',rows=rows[vi],champion=anchor,selected=output[vi])
            print(json.dumps({'outer_fold_completed':f,'selected':selected}),flush=True)
        assert np.isfinite(output).all() and np.isfinite(champout).all()
        # Outer models are a separate valid cross-fit for final all-label choice.
        # Fill other candidate columns using exactly the same outer assignments.
        for f,(ti,vi) in enumerate(outer):
            tasks=tasks_for(f'outer{f}_refit',plan['candidate_components'],rows[ti],y[ti],rows[vi],2026+f)
            completed=run_tasks(pool,tasks)
            for key,folder in completed.items():candidateout[key][vi]=load_predictions(folder,rows[vi])
    ds=[ks(y[folds==f],output[folds==f])-ks(y[folds==f],champout[folds==f]) for f in range(3)]
    delta=ks(y,output)-ks(y,champout)
    metrics={'candidate_ks':ks(y,output),'champion_ks':ks(y,champout),'delta':delta,'fold_delta':ds,
        'primary_mean_fold_delta':float(np.mean(ds)),
        'primary_metric':'Mean matched outer-fold KS; pooled OOF KS is a separate scale-sensitive diagnostic',
        'outer_selections':selections,'full_nested_selection':True,
        'historical_feature_and_shortlist_selection_reuse':True,'independent_untouched_test':False}
    np.savez_compressed(ROOT/'evidence/nested_oof.npz',rows=rows,target=y,fold=folds,prediction=output,champion=champout,**candidateout)
    final=choose(y,champout,candidateout,folds);write(ROOT/'evidence/final_inner_selection.json',final)
    metrics['final_selection']=final['selected']
    # This gate was fixed in the plan; it cannot be edited after these scores.
    metrics['promotion_gate_passed']=bool(np.mean(ds)>0 and delta>=-.003 and sum(v>0 for v in ds)>=2 and min(ds)>=-.003 and final['selected']['component'] is not None)
    write(ROOT/'benchmark_result.json',metrics)
    write(ROOT/'status.json',{'phase':'benchmark_completed','time':time.time()})
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
