"""Nested Kumo correction with 56k honest calibration rows per outer fold.

An engineered application table is the model input. No test labels, adaptive
weight search or outer-fold early stopping is used.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.special import expit,logit
from scipy.stats import ks_2samp,rankdata
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score,log_loss
import baseline as b
from gpu_backend import Engine

def read(p):return json.loads(Path(p).read_text())
def write(p,x):b.dump(x,p)
def deadline(root):
    s=read(root/'state.json')
    assert time.time()<s['deadline'] and not (root/'STOP').exists(),'Campaign deadline or stop request'
def emit(root,v):
    v={'time':time.time(),**v};print(json.dumps(v),flush=True)
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')

def metric(y,p):
    y=np.asarray(y);p=np.asarray(p);m=b.scores(y,p)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert abs(m['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-12
    n1=int(y.sum());auc=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
    assert abs(auc-m['AUC'])<1e-12
    m['logloss']=float(log_loss(y,p));return m

def load(root):
    plan=read(root/'plan.json');inp=Path(plan['input']);data=pd.read_parquet(inp/'data/train_features.parquet')
    lock=read(inp/'BASELINE_LOCK.json');definition=read(inp/'provenance/definition.json')
    assert len(data)==70000 and data.application_id.is_unique and data.customer_id.is_unique
    return plan,inp,data,lock,definition

def indices(inp,data,f):
    z=np.load(inp/f'artifacts/fold{f}/indices.npz')
    ix={k:z[k] for k in z.files};meta=read(inp/f'artifacts/fold{f}/metadata.json')
    for k,label in [('train','training'),('validation','validation'),('context','context'),('correction','correction_training')]:
        assert b.ids_hash(data,ix[k])==meta[label+'_ids_sha256']
    assert not np.intersect1d(ix['train'],ix['validation']).size
    assert not np.intersect1d(ix['context'],ix['correction']).size
    assert np.array_equal(np.sort(np.r_[ix['context'],ix['correction']]),ix['train'])
    return ix

def preflight(root):
    plan,inp,data,lock,definition=load(root);y=data.target.to_numpy(int)
    assert not set(lock['Kumo']['features']+lock['correction']['features']+lock['original_spline']['features'])&{'target','application_id','customer_id','split'}
    seen=np.zeros(len(y),int);original=np.full(len(y),np.nan);nested_checks=[]
    frozen=list(StratifiedKFold(5,shuffle=True,random_state=31027).split(np.zeros(len(y)),y))
    for f in range(5):
        ix=indices(inp,data,f);ti=ix['train'];vi=ix['validation']
        assert np.array_equal(ti,frozen[f][0]) and np.array_equal(vi,frozen[f][1])
        p=pd.read_parquet(inp/f'artifacts/fold{f}/final_oof.parquet')
        assert p.application_id.tolist()==data.application_id.iloc[vi].tolist()
        original[vi]=p.prediction;seen[vi]+=1;inner_seen=np.zeros(len(ti),int)
        inner=StratifiedKFold(10,shuffle=True,random_state=plan['inner_seed']+f)
        for j,(tr,qr) in enumerate(inner.split(np.zeros(len(ti)),y[ti])):
            available=ti[tr];ctx,_=train_test_split(available,train_size=50000,stratify=y[available],random_state=plan['inner_seed']+100*f+j)
            assert not np.intersect1d(ctx,ti[qr]).size and not np.intersect1d(ctx,vi).size
            assert len(ctx)==50000 and len(qr)==5600 and min(np.bincount(y[ti[qr]]))>500
            inner_seen[qr]+=1
        assert np.all(inner_seen==1);nested_checks.append({'fold':f,'fully_covered_inner_rows':len(ti),'context_rows':50000})
    assert np.all(seen==1)
    m=metric(y,original);assert abs(m['KS']-.5292994659005608)<1e-12 and abs(m['AUC']-.8472026883917805)<1e-12
    for pp in [np.ones(4),np.array([0,0,1,1])]:metric(np.array([0,1,0,1]),pp)
    result={'status':'conditional_methodology_pass','known_baseline':m,'nested_boundary_checks':nested_checks,
        'independent_audit':False,'point_in_time_certified':False,'broad_search_allowed':False,
        'plan_sha':b.sha(root/'plan.json'),'scope':'one fixed correction-data intervention'}
    write(root/'preflight.json',result);return result

def checked_prediction(path,context,query):
    z=np.load(path/'prediction.npz');meta=read(path/'complete.json')
    for name,h in meta['files'].items():assert b.sha(path/name)==h
    assert np.array_equal(z['context'],context) and np.array_equal(z['rows'],query)
    return z['prediction']

def gpu_prediction(root,engine,data,context,query,path,seed,outer_valid):
    deadline(root)
    assert not np.intersect1d(context,query).size
    assert not np.intersect1d(context,outer_valid).size
    if (path/'complete.json').exists():return checked_prediction(path,context,query)
    path.mkdir(parents=True,exist_ok=False)
    write(path/'boundary.json',{'context_ids_sha':b.ids_hash(data,context),'query_ids_sha':b.ids_hash(data,query),
        'context_rows':len(context),'query_rows':len(query),'seed':seed,'estimators':8,
        'estimator_batch_size':1,'query_batch_size':512,'outer_validation_labels_used':False})
    emit(root,{'stage':'GPU_fit','job':str(path.relative_to(root)),'context_rows':len(context),'query_rows':len(query)})
    start=time.time();p=engine.fit_predict(context,data.iloc[query],seed)
    assert len(p)==len(query) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    np.savez_compressed(path/'prediction.npz',context=context,rows=query,prediction=p)
    write(path/'complete.json',{'seconds':time.time()-start,'frozen_backbone':True,
        'checkpoint_sha':read(root/'plan.json')['checkpoint_sha'],
        'model_retention':'immutable shared backbone plus retained context row indices and complete input table',
        'files':{p.name:b.sha(p) for p in path.iterdir() if p.is_file()}})
    emit(root,{'stage':'GPU_fit_complete','job':str(path.relative_to(root)),'seconds':time.time()-start,
        'peak_reserved_gib':engine.torch.cuda.max_memory_reserved()/2**30})
    return p

def fold(root,f):
    plan,inp,data,lock,definition=load(root);ix=indices(inp,data,f);dest=root/f'folds/fold{f}'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    assert b.sha(plan['checkpoint'])==lock['Kumo']['checkpoint_sha256']
    engine=Engine(plan['checkpoint'],lock['Kumo']['features'],inp/'data/train_features.parquet')
    h=ix['correction'];vi=ix['validation'];ti=ix['train'];y=data.target.to_numpy(int)
    # Regenerate a matched baseline with precisely the GPU batching used here.
    raw=gpu_prediction(root,engine,data,ix['context'],np.r_[h,vi],dest/'outer_context',2026+f,vi)
    ph,pv=raw[:len(h)],raw[len(h):]
    original_model=b.refit_correction(definition,data.iloc[h],ph)
    joblib.dump(original_model,dest/'control_correction.joblib',compress=3)
    spline=joblib.load(inp/f'artifacts/fold{f}/spline.joblib');sp=b.predict_spline(spline,data.iloc[vi])
    cached_spline=pd.read_parquet(inp/f'artifacts/fold{f}/spline_oof.parquet')
    assert np.max(abs(sp-cached_spline.prediction))<1e-10
    reference=.75*b.corrected_probability(original_model,data.iloc[vi],pv)[0]+.25*sp
    historical=pd.read_parquet(inp/f'artifacts/fold{f}/final_oof.parquet').prediction.to_numpy()
    write(dest/'numerical_control.json',{'raw_max_error':float(np.max(abs(pv-pd.read_parquet(inp/f'artifacts/fold{f}/kumo_oof.parquet').prediction))),
        'combined_max_error':float(np.max(abs(reference-historical))),'new_batching_requires_matched_baseline':True})
    # Every outer-training row is predicted by a context excluding that row.
    inner=np.full(len(ti),np.nan);seen=np.zeros(len(ti),int)
    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=plan['inner_seed']+f)
    for j,(local_fit,local_query) in enumerate(skf.split(np.zeros(len(ti)),y[ti])):
        deadline(root);available=ti[local_fit];qr=ti[local_query]
        context,_=train_test_split(available,train_size=50000,stratify=y[available],random_state=plan['inner_seed']+100*f+j)
        assert not np.intersect1d(qr,vi).size and len(context)==50000
        inner[local_query]=gpu_prediction(root,engine,data,context,qr,dest/f'inner{j}',2026+f,vi)
        seen[local_query]+=1
    assert np.all(seen==1) and np.isfinite(inner).all()
    np.savez_compressed(dest/'calibration_oof.npz',rows=ti,prediction=inner)
    # One original objective and parameter set; only calibration sample changes.
    model=b.refit_correction(definition,data.iloc[ti],inner)
    joblib.dump(model,dest/'nested_correction.joblib',compress=3)
    candidate=.75*b.corrected_probability(model,data.iloc[vi],pv)[0]+.25*sp
    np.savez_compressed(dest/'validation.npz',rows=vi,target=y[vi],baseline=reference,candidate=candidate,historical=historical)
    write(dest/'complete.json',{'correction_training_rows':len(ti),'outer_evaluation_rows':len(vi),'inner_folds':10,
        'gradient_max':model['gradient_max_abs'],'outer_labels_used_to_train':False,
        'files':{p.name:b.sha(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'}})
    emit(root,{'stage':'outer_fold_complete','fold':f})

def finish(root):
    deadline(root);plan,inp,data,lock,definition=load(root);y=data.target.to_numpy(int);n=len(y)
    seen=np.zeros(n,int);folds=np.full(n,-1);ps={k:np.full(n,np.nan) for k in ['baseline','candidate','historical']}
    for f in range(5):
        d=root/f'folds/fold{f}'
        for name,h in read(d/'complete.json')['files'].items():assert b.sha(d/name)==h
        z=np.load(d/'validation.npz');rows=z['rows'];assert np.array_equal(y[rows],z['target']);seen[rows]+=1;folds[rows]=f
        for name in ps:ps[name][rows]=z[name]
    assert np.all(seen==1)
    report={name:metric(y,p) for name,p in ps.items()}
    assert abs(report['historical']['KS']-.5292994659005608)<1e-12
    assert report['baseline']['KS']>.52 and report['baseline']['AUC']>.84,'Matched baseline failed numerical qualification'
    fd=[]
    for f in range(5):
        ii=folds==f;a=metric(y[ii],ps['candidate'][ii]);b0=metric(y[ii],ps['baseline'][ii])
        fd.append({k:a[k]-b0[k] for k in ['KS','AUC','logloss']})
    delta={k:report['candidate'][k]-report['baseline'][k] for k in ['KS','AUC','logloss']}
    rng=np.random.default_rng(20261020);groups=[np.flatnonzero((folds==f)&(y==c)) for f in range(5) for c in [0,1]]
    draws=[]
    for rep in range(3000):
        if rep%100==0:deadline(root)
        rows=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);a=b.scores(y[rows],ps['candidate'][rows]);ref=b.scores(y[rows],ps['baseline'][rows])
        draws.append([a['KS']-ref['KS'],a['AUC']-ref['AUC']])
    ci=np.quantile(draws,[.00625,.99375],axis=0) # 2 objectives x 2 interventions in this invocation
    accepted=bool(delta['KS']>0 and delta['AUC']>0 and delta['logloss']<=.0005
        and sum(r['KS']>0 for r in fd)>=4 and min(r['KS'] for r in fd)>=-.003
        and sum(r['AUC']>0 for r in fd)>=3 and ci[0,0]>0 and ci[0,1]>0
        and report['candidate']['KS']>report['historical']['KS'] and report['candidate']['AUC']>report['historical']['AUC'])
    np.savez_compressed(root/'oof.npz',target=y,fold=folds,**ps)
    result={'status':'complete','accepted':accepted,'selected':'nested' if accepted else 'incumbent',
        'metrics':report,'delta_matched':delta,'fold_deltas':fd,'KS_adjusted_CI':ci[:,0].tolist(),'AUC_adjusted_CI':ci[:,1].tolist(),
        'independent_confirmation':False,'point_in_time_certified':False,'public_score_guaranteed':False,
        'bootstrap':'3000 paired fold/class resamples, four-test Bonferroni, conditional on fitted predictions; historical adaptive selection uncorrected',
        'new_kumo_context_fits':55,'correction_fits':10}
    write(root/'result.json',result)
    model=root/'model';model.mkdir()
    cols=list(dict.fromkeys(lock['Kumo']['features']+lock['correction']['features']+lock['original_spline']['features']))
    data[['application_id','target']+cols].to_parquet(model/'context_data.parquet',index=False)
    shutil.copy2(inp/'BASELINE_LOCK.json',model/'baseline_lock.json')
    for f in range(5):
        out=model/f'fold{f}';out.mkdir()
        src=root/f'folds/fold{f}/nested_correction.joblib' if accepted else inp/f'artifacts/fold{f}/correction.joblib'
        shutil.copy2(src,out/'correction.joblib');shutil.copy2(inp/f'artifacts/fold{f}/spline.joblib',out/'spline.joblib')
        shutil.copy2(inp/f'artifacts/fold{f}/indices.npz',out/'indices.npz')
    write(model/'manifest.json',{'selected':result['selected'],'checkpoint':plan['checkpoint'],'checkpoint_sha':plan['checkpoint_sha'],
        'training_policy':'retained five-fold ensemble; all permitted labels used across components; no unevaluated change to Kumo context size',
        'features':cols,'files':{str(f.relative_to(model)):b.sha(f) for f in model.rglob('*') if f.is_file()}})
    lines=['# Nested correction pipeline','',f'Selected: {result["selected"]}. New improvement accepted: {accepted}.','',
        '| Pipeline | KS | ROC-AUC | Log loss |','|---|---:|---:|---:|']
    for name,m in report.items():lines.append(f'| {name} | {m["KS"]:.6f} | {m["AUC"]:.6f} | {m["logloss"]:.6f} |')
    lines+=['','Both KS and AUC must improve under the frozen criteria. Previously exposed labels and inherited snapshot uncertainty remain. No leaderboard improvement is guaranteed.','',
        'All frozen backbones, context indices, inner OOF predictions and fitted correctors are retained. No competition submission was generated.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    emit(root,{'DONE':True,'accepted':accepted,'report':str(root/'REPORT.md')})

def predict(root,features,output):
    p=root/'model';m=read(p/'manifest.json')
    for name,h in m['files'].items():assert b.sha(p/name)==h
    assert b.sha(m['checkpoint'])==m['checkpoint_sha']
    q=pd.read_parquet(features);assert 'target' not in q and q.application_id.is_unique and set(m['features']).issubset(q.columns)
    lock=read(p/'baseline_lock.json');engine=Engine(m['checkpoint'],lock['Kumo']['features'],p/'context_data.parquet')
    out=np.zeros(len(q))
    for f in range(5):
        raw=engine.predict_fold(f,np.load(p/f'fold{f}/indices.npz')['context'],q)
        correction=joblib.load(p/f'fold{f}/correction.joblib');spline=joblib.load(p/f'fold{f}/spline.joblib')
        out+=(.75*b.corrected_probability(correction,q,raw)[0]+.25*b.predict_spline(spline,q))/5
    assert np.isfinite(out).all() and not output.exists()
    frame=pd.DataFrame({'application_id':q.application_id,'default_probability':out})
    if output.suffix=='.csv':frame.to_csv(output,index=False)
    else:frame.to_parquet(output,index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fold','finish','predict'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--fold',type=int)
    parser.add_argument('--features',type=Path);parser.add_argument('--output',type=Path);a=parser.parse_args()
    if a.action=='fold':fold(a.root,a.fold)
    elif a.action=='finish':finish(a.root)
    else:predict(a.root,a.features,a.output)
