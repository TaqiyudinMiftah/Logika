"""Fixed blends reuse valid OOF; learned blends build genuinely nested OOF."""
from __future__ import annotations
import itertools
import random
import time
from pathlib import Path
import numpy as np
from .common import read,write,sha,identity,metrics,verify_oof,json_safe
from .pipeline import fit_predict

def combine(matrix,method,weights):
    weights=np.asarray(weights,dtype=float);weights=weights/weights.sum();matrix=np.clip(matrix,1e-8,1-1e-8)
    if method in ['probability','rank']:return matrix@weights
    if method=='logit':
        z=np.log(matrix/(1-matrix))@weights;return 1/(1+np.exp(-np.clip(z,-30,30)))
    if method=='geometric':return np.exp(np.log(matrix)@weights)
    if method=='trimmed':return np.mean(np.sort(matrix,axis=1)[:,1:-1],axis=1)
    raise ValueError(method)

def rank_values(p,cdf):
    # CDF comes from model-training predictions, never outer-validation labels/data.
    knots=np.asarray(cdf);unique,idx=np.unique(knots,return_index=True)
    return np.interp(p,unique,np.linspace(0,1,len(knots))[idx],left=0,right=1)

def fixed(root,bank,parents,limit,until):
    root=Path(root);arrays={};ranks={};scores={}
    for tid in parents:
        folder=root/'runs'/tid;complete=read(folder/'complete.json')
        for file,h in complete['files'].items():
            if sha(folder/file)!=h:raise ValueError('Invalid ensemble parent '+tid)
        p=np.load(folder/'prediction.npy');verify_oof(bank.y,p,bank.folds);arrays[tid]=p;scores[tid]=read(folder/'scores.json')
        cf=read(folder/'cdf.json');rank=np.empty_like(p)
        for f in range(3):
            vi=bank.folds==f;rank[vi]=rank_values(p[vi],cf[str(f)])
        ranks[tid]=rank
    designs=[]
    for pair in itertools.combinations(parents,2):
        for method in ['probability','rank','logit','geometric']:
            for w in [.1,.25,.5,.75,.9]:designs.append({'parents':list(pair),'weights':[w,1-w],'method':method})
    for size in [3,4,5]:
        for group in itertools.islice(itertools.combinations(parents,size),512):
            for method in ['probability','rank','logit','geometric','trimmed']:
                designs.append({'parents':list(group),'weights':[1/size]*size,'method':method})
    random.Random(20261016).shuffle(designs)
    # Equal probability blends of every pair are evaluated first as interpretable controls.
    designs=[{'parents':list(p),'weights':[.5,.5],'method':'probability'} for p in itertools.combinations(parents,2)]+designs
    seen=set();out=[]
    for d in designs:
        key=identity(d)
        if key in seen:continue
        seen.add(key)
        if len(out)>=limit or time.time()>until:break
        tid='e_'+key[:20];folder=root/'runs'/tid;folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():out.append(tid);continue
        a=np.column_stack([(ranks if d['method']=='rank' else arrays)[p] for p in d['parents']])
        p=combine(a,d['method'],d['weights']);score=verify_oof(bank.y,p,bank.folds)
        write(folder/'task.json',{'id':tid,'mode':'fixed_ensemble','recipe':d})
        write(folder/'identity.json',{'bank_id':bank.manifest['bank_id'],'parents':{v:sha(root/'runs'/v/'complete.json') for v in d['parents']},'recipe':d})
        np.save(folder/'prediction.npy',p)
        np.savez_compressed(folder/'oof.npz',application_id=bank.metadata.application_id.iloc[bank.dev].to_numpy(dtype=str),target=bank.y,fold=bank.folds,prediction=p)
        write(folder/'scores.json',{**score,'state':'completed','kind':'fixed_ensemble','recipe':d,'bank_id':bank.manifest['bank_id'],'code_hash':read(root/'campaign.json')['code_hash'],
                                  'prediction_correlations':json_safe(np.corrcoef(a,rowvar=False).tolist())})
        write(folder/'complete.json',{'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name not in ['complete.json','status.json']}})
        write(folder/'status.json',{'state':'completed'});out.append(tid)
    return out

def learn_weights(x,y,seed=2026):
    rng=np.random.default_rng(seed);choices=[np.ones(x.shape[1])/x.shape[1],*np.eye(x.shape[1]),*rng.dirichlet(np.ones(x.shape[1])*2,size=64)]
    return max(choices,key=lambda w:metrics(y,x@w)['ks'])

def nested(root,bank,task,folder):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    bases=task['recipe']['bases'];method=task['recipe']['method'];p=np.full(len(bank.y),np.nan);details=[]
    for f in range(3):
        ti=np.flatnonzero(bank.folds!=f);vi=np.flatnonzero(bank.folds==f);y=bank.y[ti]
        inner=np.full((len(ti),len(bases)),np.nan);outer=np.zeros((len(vi),len(bases)))
        for j,recipe in enumerate(bases):
            device=task.get('device','cpu') if recipe['model'] in ['cat','xgb'] else 'cpu'
            for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=6000+f).split(ti,y)):
                inner[b,j],*_=fit_predict(bank,recipe,bank.dev[ti[a]],y[a],bank.dev[ti[b]],device,7000+f*10+k)
            outer[:,j],*_=fit_predict(bank,recipe,bank.dev[ti],y,bank.dev[vi],device,8000+f)
        if not np.isfinite(inner).all():raise ValueError('Incomplete inner OOF')
        if method=='logistic':
            meta=LogisticRegression(C=.1,max_iter=200);meta.fit(inner,y);p[vi]=meta.predict_proba(outer)[:,1];parameters=meta.coef_.tolist()
        elif method=='weights':
            w=learn_weights(inner,y,9000+f);p[vi]=outer@w;parameters=w.tolist()
        else:
            col=bank.lookup['m_repayment_all_count'];gt=np.nan_to_num(bank.numeric[bank.dev[ti],col])>0;gv=np.nan_to_num(bank.numeric[bank.dev[vi],col])>0
            w=learn_weights(inner,y,9000+f);parameters={}
            for group in [False,True]:
                ix=gt==group;wg=learn_weights(inner[ix],y[ix],9100+f) if ix.sum()>=200 and sum(y[ix])>=20 and sum(1-y[ix])>=20 else w
                p[vi[gv==group]]=outer[gv==group]@wg;parameters[str(group)]=wg.tolist()
        np.savez_compressed(folder/f'inner_fold_{f}.npz',outer_training_rows=bank.dev[ti],inner_oof=inner,target=y,outer_validation_rows=bank.dev[vi],outer_predictions=outer)
        details.append({'fold':f,'outer_train_hash':identity(bank.dev[ti].tolist()),'parameters':parameters})
        write(folder/'nested_audit.json',details)
    np.save(folder/'prediction.npy',p)
    np.savez_compressed(folder/'oof.npz',application_id=bank.metadata.application_id.iloc[bank.dev].to_numpy(dtype=str),fold=bank.folds,target=bank.y,prediction=p)
    return {'kind':'nested_ensemble','recipe':task['recipe'],**verify_oof(bank.y,p,bank.folds)}

def confirm(root,bank,task,folder):
    if task['parent_id'] not in read(root/'finalists.json')['ids']:raise ValueError('Unfrozen ensemble')
    spec=task['recipe'];pcols=[]
    if 'bases' in spec:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        rows=np.load(bank.root/'confirmation_rows.npy');inner=np.zeros((len(bank.y),len(spec['bases'])));query=np.zeros((len(rows),len(spec['bases'])))
        for j,recipe in enumerate(spec['bases']):
            device=task.get('device','cpu') if recipe['model'] in ['cat','xgb'] else 'cpu'
            for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=6000).split(bank.dev,bank.y)):
                inner[b,j],*_=fit_predict(bank,recipe,bank.dev[a],bank.y[a],bank.dev[b],device,7000+k)
            query[:,j],pre,base_model,weight,quantiles,messages=fit_predict(bank,recipe,bank.dev,bank.y,rows,device,8000)
            import joblib
            pre.bank=None
            joblib.dump({'preprocessing':pre,'model':base_model,'weight':weight,'recipe':recipe,'seed':8000},folder/f'base_{j}.joblib',compress=1)
            write(folder/f'base_{j}_preprocessing.json',pre.audit)
        if spec['method']=='logistic':
            model=LogisticRegression(C=.1,max_iter=200).fit(inner,bank.y);p=model.predict_proba(query)[:,1]
            joblib.dump(model,folder/'meta_model.joblib',compress=1)
        elif spec['method']=='weights':
            weights=learn_weights(inner,bank.y);p=query@weights;write(folder/'learned_weights.json',weights.tolist())
        else:
            col=bank.lookup['m_repayment_all_count'];gt=np.nan_to_num(bank.numeric[bank.dev,col])>0;gv=np.nan_to_num(bank.numeric[rows,col])>0;p=np.zeros(len(rows));w=learn_weights(inner,bank.y)
            routing={}
            for group in [False,True]:
                ix=gt==group;wg=learn_weights(inner[ix],bank.y[ix]) if ix.sum()>=200 and sum(bank.y[ix])>=20 and sum(1-bank.y[ix])>=20 else w;p[gv==group]=query[gv==group]@wg;routing[str(group)]=wg.tolist()
            write(folder/'routing_weights.json',routing)
    else:
        for parent in spec['parents']:
            child=root/'runs'/('c_'+parent)
            if not (child/'complete.json').exists():raise ValueError('Missing confirmation component '+parent)
            pp=np.load(child/'prediction.npy')
            if spec['method']=='rank':pp=rank_values(pp,read(child/'scores.json')['train_quantiles'])
            pcols.append(pp)
        p=combine(np.column_stack(pcols),spec['method'],spec['weights'])
    np.save(folder/'prediction.npy',p);y=np.load(bank.root/'confirmation_y.npy')
    np.savez_compressed(folder/'oof.npz',target=y,prediction=p,rows=np.load(bank.root/'confirmation_rows.npy'))
    return {'kind':'confirmation','parent_id':task['parent_id'],'metrics':metrics(y,p)}
