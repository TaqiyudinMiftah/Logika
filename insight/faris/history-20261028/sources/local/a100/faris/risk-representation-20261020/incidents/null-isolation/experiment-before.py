"""Nested teacher features for one frozen-Kumo research hypothesis."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse,copy,gc,json,time,traceback,shutil
from concurrent.futures import ThreadPoolExecutor,as_completed
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss,average_precision_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import baseline as b
from metaexp.features import Bank
from metaexp.pipeline import fit_predict,predict
from gpu_backend import Engine

HERE=Path(__file__).resolve().parent
EXTRA=['learned_horizontal_risk_logit','learned_horizontal_disagreement']

def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):b.dump(obj,path)
def event(root,stage,**values):
    obj=dict(time=time.time(),stage=stage,**values)
    print(json.dumps(obj),flush=True)
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
def deadline(root):
    s=read(root/'state.json')
    assert time.time()<s['deadline'] and not (root/'STOP').exists(),'Deadline or stop request'
def metric(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    m=b.scores(y,p);m['logloss']=float(log_loss(y,p));m['average_precision']=float(average_precision_score(y,p))
    assert abs(m['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-12
    n1=y.sum();auc=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
    assert abs(m['AUC']-auc)<1e-12
    return m
def load(root):
    plan=read(root/'plan.json');inp=Path(plan['input'])
    data=pd.read_parquet(inp/'data/train_features.parquet')
    assert len(data)==70000 and data.application_id.is_unique and data.customer_id.is_unique
    bank=Bank(Path(plan['bank']).parent)
    assert data.application_id.tolist()==bank.metadata.application_id.iloc[:len(data)].tolist()
    assert data.customer_id.tolist()==bank.metadata.customer_id.iloc[:len(data)].tolist()
    lock=read(inp/'BASELINE_LOCK.json');definition=read(inp/'provenance/definition.json')
    return plan,inp,data,bank,lock,definition
def indices(inp,data,f):
    ix=dict(np.load(inp/f'artifacts/fold{f}/indices.npz'))
    meta=read(inp/f'artifacts/fold{f}/metadata.json')
    for name,label in [('train','training'),('context','context'),('correction','correction_training'),('validation','validation')]:
        assert b.ids_hash(data,ix[name])==meta[label+'_ids_sha256']
    assert len(ix['context'])==50000 and len(ix['correction'])==6000 and len(ix['validation'])==14000
    assert np.array_equal(np.sort(np.r_[ix['context'],ix['correction']]),ix['train'])
    assert not np.intersect1d(ix['context'],ix['correction']).size
    assert not np.intersect1d(ix['train'],ix['validation']).size
    return ix
def saved_reference(plan,inp,data,f,ix):
    ref=Path(plan['matched_reference'])/f'folds/fold{f}'
    q=np.r_[ix['correction'],ix['validation']]
    context=ref/'outer_context';meta=read(context/'complete.json')
    for name,digest in meta['files'].items():assert b.sha(context/name)==digest
    raw=np.load(context/'prediction.npz')
    assert np.array_equal(raw['context'],ix['context']) and np.array_equal(raw['rows'],q)
    z=np.load(ref/'validation.npz')
    assert np.array_equal(z['rows'],ix['validation'])
    correction=joblib.load(ref/'control_correction.joblib')
    sp=pd.read_parquet(inp/f'artifacts/fold{f}/spline_oof.parquet')
    assert sp.application_id.tolist()==data.application_id.iloc[ix['validation']].tolist()
    replay=.75*b.corrected_probability(correction,data.iloc[ix['validation']],raw['prediction'][6000:])[0]+.25*sp.prediction.to_numpy()
    assert np.max(abs(replay-z['baseline']))<1e-12
    return z['baseline'],sp.prediction.to_numpy(),raw['prediction']
def verified_teacher(folder,rows):
    meta=read(folder/'complete.json')
    for name,digest in meta['files'].items():assert b.sha(folder/name)==digest
    z=np.load(folder/'prediction.npz');assert np.array_equal(z['rows'],rows)
    return z['prediction']
def teacher(root,bank,data,recipe,fitrows,queryrows,folder,seed):
    deadline(root)
    assert not np.intersect1d(fitrows,queryrows).size
    if (folder/'complete.json').exists():return verified_teacher(folder,queryrows)
    if folder.exists():
        # Preserve any interrupted attempt rather than silently replacing its models.
        failed=folder.with_name(folder.name+f'.partial.{int(time.time())}')
        folder.rename(failed)
    folder.mkdir(parents=True)
    dump(folder/'boundary.json',dict(fit_rows=len(fitrows),query_rows=len(queryrows),
        fit_ids_sha=b.ids_hash(data,fitrows),query_ids_sha=b.ids_hash(data,queryrows),
        fit_labels_sha=b.fingerprint(data.iloc[fitrows],['target']),
        query_labels_used=False,seed=seed))
    y=data.target.iloc[fitrows].to_numpy(int)
    start=time.time()
    probability,pre,model,weight,cdf,messages=fit_predict(bank,copy.deepcopy(recipe),fitrows,y,queryrows,'cpu',seed)
    assert probability.shape==(len(queryrows),) and np.isfinite(probability).all()
    forbidden={'target','application_id','customer_id','loan_id','order_id','split'}
    assert not set(pre.feature_names)&forbidden
    repeated=predict(model,pre.transform(queryrows[:64]),weight)
    assert np.max(abs(repeated-probability[:64]))<1e-12
    # Keep every fitted transform; reattach its immutable bank only for replay.
    pre.bank=None
    joblib.dump(dict(model=model,preprocessing=pre,weight=weight,recipe=recipe,
                     bank=bank.root.as_posix()),folder/'model.joblib',compress=3)
    restored=joblib.load(folder/'model.joblib');restored['preprocessing'].bank=bank
    after=predict(restored['model'],restored['preprocessing'].transform(queryrows[:64]),weight)
    assert np.max(abs(after-probability[:64]))<1e-12
    np.savez_compressed(folder/'prediction.npz',rows=queryrows,prediction=probability)
    dump(folder/'audit.json',dict(preprocessing=pre.audit,warnings=messages,
                                 final_columns=pre.feature_names,training_only=True))
    dump(folder/'complete.json',dict(seconds=time.time()-start,
        files={p.name:b.sha(p) for p in folder.iterdir() if p.is_file()}))
    event(root,'teacher_complete',job=str(folder.relative_to(root)),seconds=time.time()-start)
    return probability
def teacher_block(root,bank,data,recipes,fitrows,queryrows,directory,seed,workers):
    results=np.full((len(queryrows),5),np.nan)
    # Shared bank is read-only; each learner owns its preprocessor and fitted arrays.
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(teacher,root,bank,data,r['recipe'],fitrows,queryrows,
                           directory/f'teacher{i}',seed):i for i,r in enumerate(recipes)}
        for future in as_completed(tasks):results[:,tasks[future]]=future.result()
    assert np.isfinite(results).all()
    return results
def distilled(p):
    assert p.ndim==2 and p.shape[1]==5 and np.isfinite(p).all()
    trimmed=np.sort(p,axis=1)[:,1:4].mean(axis=1)
    z=logit(np.clip(p,1e-6,1-1e-6))
    features=np.column_stack([logit(np.clip(trimmed,1e-6,1-1e-6)),z.std(axis=1)])
    return features.astype(np.float32),trimmed

def preflight(root):
    plan,inp,data,bank,lock,definition=load(root)
    assert b.sha(plan['checkpoint'])==plan['checkpoint_sha']
    recipes=read(HERE/'champion.json');assert len(recipes)==5
    seen=np.zeros(len(data),int);reference=np.full(len(data),np.nan)
    for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=31027).split(data,data.target)):
        ix=indices(inp,data,f);assert np.array_equal(ix['train'],tr) and np.array_equal(ix['validation'],va)
        p,_,_=saved_reference(plan,inp,data,f,ix);reference[va]=p;seen[va]+=1
        inner_seen=np.zeros(50000,int)
        for it,iv in StratifiedKFold(5,shuffle=True,random_state=plan['inner_seed']+f).split(np.zeros(50000),data.target.iloc[ix['context']]):
            assert len(it)==40000 and len(iv)==10000
            assert not np.intersect1d(ix['context'][it],ix['context'][iv]).size
            assert not np.intersect1d(ix['context'][it],ix['validation']).size
            assert not np.intersect1d(ix['context'][it],ix['correction']).size
            inner_seen[iv]+=1
        assert np.all(inner_seen==1)
    assert np.all(seen==1)
    m=metric(data.target.to_numpy(),reference)
    assert abs(m['KS']-.5292498733857243)<1e-12 and abs(m['AUC']-.8472013286446958)<1e-12
    # Small real-data replay for all five teacher feature factories.
    ix=indices(inp,data,0);ctx=ix['context']
    smoke=teacher_block(root,bank,data,recipes,ctx[:2000],ctx[2000:2100],root/'smoke/teachers',2026,1)
    extra,_=distilled(smoke);assert extra.shape==(100,2)
    # A GPU smoke uses eligible context/query rows; new numerical columns are
    # synthetic bounded inputs here, not performance evidence.
    tiny=data.iloc[np.r_[ctx[:512],ctx[2000:2064]]].copy().reset_index(drop=True)
    for i,col in enumerate(EXTRA):tiny[col]=np.random.default_rng(901+i).normal(size=len(tiny)).astype(np.float32)
    engine=Engine(plan['checkpoint'],lock['Kumo']['features']+EXTRA,tiny)
    cache=root/'smoke/fitted_context.pt'
    pred=engine.fit_predict(np.arange(512),tiny.iloc[512:].drop(columns='target'),2026,cache_path=cache)
    replay=engine.replay(cache,tiny.iloc[512:].drop(columns='target'))
    assert np.max(abs(pred-replay))<1e-6
    assert pred.shape==(64,) and np.isfinite(pred).all()
    import torch
    accelerator=dict(name=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated(),
                     actual_predictions=len(pred),device_count=torch.cuda.device_count())
    del engine;gc.collect();torch.cuda.empty_cache()
    result=dict(status='passed',matched_baseline=m,complete_outer_assignments=True,
                teacher_context_OOF_boundaries=True,teacher_model_replay=True,accelerator=accelerator,
                smoke_teacher_fits=5,smoke_context_fits=1,historical_labels_reused=True,
                source_global_unsupervised_references_inherited=True)
    dump(root/'preflight.json',result);event(root,'preflight_pass',accelerator=accelerator)

def fold(root,f):
    deadline(root);plan,inp,data,bank,lock,definition=load(root)
    ix=indices(inp,data,f);context=ix['context'];query=np.r_[ix['correction'],ix['validation']]
    directory=root/f'folds/fold{f}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():
        for name,digest in read(directory/'complete.json')['files'].items():assert b.sha(directory/name)==digest
        return
    recipes=read(HERE/'champion.json')
    inner=StratifiedKFold(5,shuffle=True,random_state=plan['inner_seed']+f)
    context_probability=np.full((50000,5),np.nan);covered=np.zeros(50000,int)
    for j,(it,iv) in enumerate(inner.split(np.zeros(50000),data.target.iloc[context])):
        deadline(root)
        context_probability[iv]=teacher_block(root,bank,data,recipes,context[it],context[iv],
            directory/f'teachers/inner{j}',2026+f,plan['teacher_workers'])
        covered[iv]+=1
        np.savez_compressed(directory/f'inner{j}_indices.npz',fit=context[it],query=context[iv])
    assert np.all(covered==1)
    query_probability=teacher_block(root,bank,data,recipes,context,query,
        directory/'teachers/full',2026+f,plan['teacher_workers'])
    context_features,context_trimmed=distilled(context_probability)
    query_features,query_trimmed=distilled(query_probability)
    np.savez_compressed(directory/'teacher_features.npz',context=context,query=query,
        context_predictions=context_probability,query_predictions=query_probability,
        context_features=context_features,query_features=query_features,
        context_trimmed=context_trimmed,query_trimmed=query_trimmed)
    rng=np.random.default_rng(73100+f)
    perm=rng.permutation(len(context));null_sample=rng.integers(0,len(context),len(query))
    np.savez_compressed(directory/'null_indices.npz',context_permutation=perm,query_sample=null_sample)
    original,sp,raw=saved_reference(plan,inp,data,f,ix)
    results={};correctors={}
    for variant in plan['gpu_variants']:
        deadline(root)
        dest=directory/variant;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            meta=read(dest/'complete.json')
            for name,digest in meta['files'].items():assert b.sha(dest/name)==digest
            results[variant]=np.load(dest/'prediction.npz')['final'];continue
        cf=context_features if variant=='signal' else context_features[perm]
        qf=query_features if variant=='signal' else context_features[null_sample]
        context_frame=data.copy()
        query_frame=data.iloc[query].drop(columns='target').copy()
        for j,col in enumerate(EXTRA):
            context_frame[col]=np.nan
            context_frame.loc[context,col]=cf[:,j]
            query_frame[col]=qf[:,j]
        event(root,'Kumo_fit',outer_fold=f,variant=variant,context_rows=len(context),query_rows=len(query),features=92)
        engine=Engine(plan['checkpoint'],lock['Kumo']['features']+EXTRA,context_frame)
        import torch
        torch.cuda.reset_peak_memory_stats()
        assert shutil.disk_usage(root).free>16*1024**3,'Keep filesystem headroom before caching'
        start=time.time();p=engine.fit_predict(context,query_frame,2026+f,cache_path=dest/'fitted_context.pt')
        accelerator=dict(name=torch.cuda.get_device_name(),physical_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         peak_vram_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.time()-start)
        head=b.refit_correction(definition,data.iloc[ix['correction']],p[:6000])
        final=.75*b.corrected_probability(head,data.iloc[ix['validation']],p[6000:])[0]+.25*sp
        joblib.dump(head,dest/'correction.joblib',compress=3)
        repeated=.75*b.corrected_probability(joblib.load(dest/'correction.joblib'),
            data.iloc[ix['validation']].drop(columns='target'),p[6000:])[0]+.25*sp
        assert np.array_equal(final,repeated)
        np.savez_compressed(dest/'prediction.npz',rows=query,raw=p,final=final)
        dump(dest/'boundary.json',dict(context_ids_sha=b.ids_hash(data,context),
            correction_ids_sha=b.ids_hash(data,ix['correction']),validation_ids_sha=b.ids_hash(data,ix['validation']),
            teacher_fit_labels_from_context_only=True,context_features_cross_fitted=True,
            outer_labels_fitted=False,seed=2026+f,columns=lock['Kumo']['features']+EXTRA,accelerator=accelerator))
        dump(dest/'complete.json',dict(files={p.name:b.sha(p) for p in dest.iterdir() if p.is_file()}))
        results[variant]=final
        del engine,context_frame,query_frame;gc.collect();torch.cuda.empty_cache()
        event(root,'Kumo_complete',outer_fold=f,variant=variant,accelerator=accelerator)
    np.savez_compressed(directory/'validation.npz',rows=ix['validation'],baseline=original,
                        historical=pd.read_parquet(inp/f'artifacts/fold{f}/final_oof.parquet').prediction.to_numpy(),
                        teacher=query_trimmed[6000:],**results)
    dump(directory/'complete.json',dict(teacher_fits=30,kumo_context_fits=2,
        teacher_context_OOF=metric(data.target.iloc[context].to_numpy(),context_trimmed),
        files={str(p.relative_to(directory)):b.sha(p) for p in directory.rglob('*')
               if p.is_file() and p.name not in ['complete.json']}))
    event(root,'outer_fold_complete',fold=f)

def finish(root):
    deadline(root);plan,inp,data,bank,lock,definition=load(root);y=data.target.to_numpy(int)
    names=['baseline','historical','teacher','signal','null'];preds={k:np.full(len(y),np.nan) for k in names}
    fold_ids=np.full(len(y),-1);seen=np.zeros(len(y),int);hashes=0
    for f in range(5):
        d=root/f'folds/fold{f}';meta=read(d/'complete.json')
        for name,digest in meta['files'].items():assert b.sha(d/name)==digest;hashes+=1
        z=np.load(d/'validation.npz');rows=z['rows'];seen[rows]+=1;fold_ids[rows]=f
        assert np.array_equal(rows,indices(inp,data,f)['validation'])
        for name in names:preds[name][rows]=z[name]
    assert np.all(seen==1)
    metrics={name:metric(y,p) for name,p in preds.items()}
    folds={name:[metric(y[fold_ids==f],p[fold_ids==f]) for f in range(5)] for name,p in preds.items()}
    assert abs(metrics['historical']['KS']-.5292994659005608)<1e-12
    comparisons={name:{k:metrics['signal'][k]-metrics[name][k] for k in ['KS','AUC','logloss','average_precision']}
                 for name in ['baseline','null']}
    structures={}
    for name in ['baseline','null','signal']:
        p=preds[name];order=np.argsort(p,kind='stable');end=np.r_[p[order][1:]!=p[order][:-1],True]
        structures[name]=order,end
    groups=[np.flatnonzero((fold_ids==f)&(y==c)) for f in range(5) for c in [0,1]]
    rng=np.random.default_rng(20261026);draws={k:[] for k in comparisons}
    for repeat in range(plan['bootstrap_repetitions']):
        if repeat%200==0:deadline(root);event(root,'bootstrap',completed=repeat,total=plan['bootstrap_repetitions'])
        w=np.zeros(len(y),int)
        for group in groups:w[group]=rng.multinomial(len(group),np.full(len(group),1/len(group)))
        n1=w[y==1].sum();n0=w[y==0].sum();m={}
        for name,(order,end) in structures.items():
            cp=np.cumsum(w[order]*(y[order]==1))[end];cn=np.cumsum(w[order]*(y[order]==0))[end]
            dp=np.diff(np.r_[0,cp]);dn=np.diff(np.r_[0,cn])
            m[name]=np.array([np.max(abs(cp/n1-cn/n0)),np.sum(dp*(cn-.5*dn))/(n1*n0)])
        for name in draws:draws[name].append(m['signal']-m[name])
    intervals={k:np.quantile(v,[.00625,.99375],axis=0).tolist() for k,v in draws.items()}
    point_goal=bool(metrics['signal']['KS']>=.54 and metrics['signal']['AUC']>=metrics['historical']['AUC'])
    evidence_positive=bool(all(comparisons[k]['KS']>0 and comparisons[k]['AUC']>0 for k in comparisons))
    result=dict(status='complete',metrics=metrics,fold_metrics=folds,comparisons=comparisons,
        adjusted_delta_intervals=intervals,interval_columns=['KS','AUC'],
        interval_method='2000 paired fold/class bootstrap, Bonferroni four comparisons/metrics; conditional on fitted predictions, excludes historical selection and retraining uncertainty',
        point_goal_reached=point_goal,evidence_positive=evidence_positive,teacher_fits=150,kumo_context_fits=10,
        hashes_verified=hashes,OOF_rows=70000,external_confirmation=False,point_in_time_certified=False,
        historical_label_reuse=True,submission_generated=False)
    dump(root/'result.json',result)
    np.savez_compressed(root/'oof.npz',target=y,fold=fold_ids,**preds)
    np.savez_compressed(root/'bootstrap.npz',**{k:np.asarray(v) for k,v in draws.items()})
    lines=['# Learned risk representation results','',f'Point target .54 reached with nondecreasing AUC: {point_goal}.',
           f'KS and AUC both exceed both matched controls: {evidence_positive}.','',
           '| Pipeline | KS | AUC | Log loss |','|---|---:|---:|---:|']
    for name,m in metrics.items():lines.append(f'| {name} | {m["KS"]:.8f} | {m["AUC"]:.8f} | {m["logloss"]:.8f} |')
    lines+=['','All five outer folds and all 150 teachers completed; query labels never trained context features.',
            'All trained models, transformations, context inputs and checkpoint dependencies are retained.',
            'The null augmentation checks width/permutation effects. The historical labels have repeated prior exposure.',
            'No new competition submission was created. Inspect paired intervals and per-fold changes before any promotion.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    event(root,'DONE',status='complete',point_goal_reached=point_goal,evidence_positive=evidence_positive)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','fold','finish'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--fold',type=int);a=p.parse_args()
    if a.action=='preflight':preflight(a.root)
    elif a.action=='fold':fold(a.root,a.fold)
    else:finish(a.root)
