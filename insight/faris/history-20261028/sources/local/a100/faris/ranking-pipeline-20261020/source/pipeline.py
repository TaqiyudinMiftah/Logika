"""Preserved Kumo + spline pipeline with one controlled ranking-loss intervention."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import argparse,copy,hashlib,json,shutil,sys,time,traceback
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.optimize import minimize,check_grad
from scipy.special import expit,logit
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import roc_auc_score,log_loss,average_precision_score
import baseline as base

def read(p):return json.loads(Path(p).read_text())
def write(p,x):base.dump(x,p)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def check_time(root):
    s=read(Path(root)/'state.json')
    assert time.time()<s['deadline'],'Absolute computation deadline exceeded'

def measure(y,p):
    v=base.scores(y,p);v.update(logloss=float(log_loss(y,p)),AP=float(average_precision_score(y,p)))
    assert abs(v['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-12
    n1=int(y.sum());a=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
    assert abs(v['AUC']-a)<1e-12
    return v

def objective(beta,x,offset,y,pos,neg,alpha=.25,penalty=.003):
    eta=offset+x@beta
    loss=(1-alpha)*np.mean(np.logaddexp(0,eta)-y*eta)
    derivative=(1-alpha)*(expit(eta)-y)/len(y)
    margin=eta[pos]-eta[neg]
    loss+=alpha*np.mean(np.logaddexp(0,-margin))
    pair_grad=-alpha*expit(-margin)/len(pos)
    derivative+=np.bincount(pos,weights=pair_grad,minlength=len(y))-np.bincount(neg,weights=pair_grad,minlength=len(y))
    loss+=.5*penalty*(beta[1:]@beta[1:])
    grad=x.T@derivative;grad[1:]+=penalty*beta[1:]
    return float(loss),grad

def fit_rank(bundle,frame,raw,seed,root):
    config=read(root/'plan.json');y=frame.target.to_numpy(int)
    x,off=base.design(bundle,frame,raw)
    rng=np.random.default_rng(seed);pos=rng.choice(np.flatnonzero(y==1),config['pairs'],replace=True)
    neg=rng.choice(np.flatnonzero(y==0),config['pairs'],replace=True)
    def fun(beta):
        check_time(root)
        return objective(beta,x,off,y,pos,neg,config['rank_weight'],config['penalty'])
    result=minimize(fun,bundle['coefficients'],method='L-BFGS-B',jac=True,
        options={'maxiter':500,'ftol':1e-12,'gtol':1e-7})
    value,grad=fun(result.x)
    assert result.success and np.max(abs(grad))<1e-5,(result.message,np.max(abs(grad)))
    new=copy.deepcopy(bundle);new.update(coefficients=result.x,objective=value,
        gradient_max_abs=float(np.max(abs(grad))),iterations=int(result.nit),
        training_objective='0.75 BCE + 0.25 sampled pairwise logistic + L2',
        ranking_seed=seed,pair_sha256=hashlib.sha256(np.stack([pos,neg]).tobytes()).hexdigest())
    return new

def inputs(root,fold):
    inp=root/'input';train=pd.read_parquet(inp/'data/train_features.parquet')
    d=inp/f'artifacts/fold{fold}'
    with np.load(d/'indices.npz') as z:idx={k:z[k] for k in z.files}
    for a,b in [('context','correction'),('train','validation')]:
        assert not np.intersect1d(idx[a],idx[b]).size
    assert np.array_equal(np.sort(np.r_[idx['context'],idx['correction']]),idx['train'])
    h=train.iloc[idx['correction']];v=train.iloc[idx['validation']]
    rawh=pd.read_parquet(d/'kumo_holdout.parquet');rawv=pd.read_parquet(d/'kumo_oof.parquet')
    assert rawh.application_id.tolist()==h.application_id.tolist()
    assert rawv.application_id.tolist()==v.application_id.tolist()
    assert np.array_equal(rawh.target,h.target) and np.array_equal(rawv.target,v.target)
    meta=read(d/'metadata.json')
    for name,key in [('context','context'),('correction','correction_training'),('train','training'),('validation','validation')]:
        assert base.ids_hash(train,idx[name])==meta[key+'_ids_sha256']
    return train,d,idx,h,v,rawh.prediction.to_numpy(),rawv.prediction.to_numpy()

def fold_train(root,fold):
    check_time(root);out=root/f'fits/fold{fold}';out.mkdir(parents=True,exist_ok=False)
    train,d,idx,h,v,ph,pv=inputs(root,fold);plan=read(root/'plan.json')
    print(json.dumps({'fold':fold,'stage':'baseline_reproduction'}),flush=True)
    original=joblib.load(d/'correction.joblib');spline=joblib.load(d/'spline.joblib')
    sp=base.predict_spline(spline,v);cached=pd.read_parquet(d/'spline_oof.parquet')
    assert cached.application_id.tolist()==v.application_id.tolist()
    assert np.max(abs(sp-cached.prediction))<1e-10
    p0=.75*base.corrected_probability(original,v,pv)[0]+.25*sp
    expected=pd.read_parquet(d/'final_oof.parquet')
    assert expected.application_id.tolist()==v.application_id.tolist()
    assert np.max(abs(p0-expected.prediction))<1e-10
    # Fit baseline first with precisely the original 6k correction rows.
    definition=read(root/'input/provenance/definition.json')
    fitted=base.refit_correction(definition,h,ph)
    control=.75*base.corrected_probability(fitted,v,pv)[0]+.25*sp
    reproduction=float(np.max(abs(control-p0)))
    assert reproduction<2e-6,('Baseline refit drift',reproduction)
    joblib.dump(fitted,out/'baseline_refit.joblib',compress=3)
    predictions={'baseline':p0,'control_refit':control}
    for seed in plan['ranking_seeds']:
        check_time(root);model=fit_rank(fitted,h,ph,seed+fold,root)
        predictions[f'rank_s{seed}']=.75*base.corrected_probability(model,v,pv)[0]+.25*sp
        joblib.dump(model,out/f'rank_s{seed}.joblib',compress=3)
        print(json.dumps({'fold':fold,'ranking_seed':seed,'iterations':model['iterations'],'stage':'fitted'}),flush=True)
    np.savez_compressed(out/'predictions.npz',rows=idx['validation'],target=v.target.to_numpy(int),**predictions)
    write(out/'boundary.json',{'context_rows':50000,'correction_rows':6000,'validation_rows':14000,
        'correction_ids_sha':base.ids_hash(train,idx['correction']),
        'evaluation_ids_sha':base.ids_hash(train,idx['validation']),
        'encoder_fit_partition':'correction','upstream_kumo_fit_partition':'context',
        'original_spline_fit_partition':'train','evaluation_labels_used_to_fit':False,
        'baseline_reproduction_max_error':reproduction,'plan_sha':base.sha(root/'plan.json')})
    write(out/'complete.json',{'files':{p.name:base.sha(p) for p in out.iterdir() if p.is_file()}})

def evaluate(root):
    check_time(root);plan=read(root/'plan.json');train=pd.read_parquet(root/'input/data/train_features.parquet')
    n=len(train);y=train.target.to_numpy(int);seen=np.zeros(n,int);fold=np.full(n,-1);pred={}
    for f in range(5):
        d=root/f'fits/fold{f}'
        for name,h in read(d/'complete.json')['files'].items():assert base.sha(d/name)==h
        z=np.load(d/'predictions.npz');rows=z['rows'];assert np.array_equal(z['target'],y[rows])
        seen[rows]+=1;fold[rows]=f
        for name in z.files:
            if name in ['rows','target']:continue
            if name not in pred:pred[name]=np.full(n,np.nan)
            pred[name][rows]=z[name]
    assert np.all(seen==1) and all(np.isfinite(v).all() for v in pred.values())
    np.savez_compressed(root/'oof.npz',target=y,fold=fold,**pred)
    reference=pred['baseline'];reference_metrics=measure(y,reference)
    assert abs(reference_metrics['KS']-.5292994659005608)<1e-12
    assert abs(reference_metrics['AUC']-.8472026883917805)<1e-12
    records=[]
    for name,p in pred.items():
        m=measure(y,p);delta={k:m[k]-reference_metrics[k] for k in ['KS','AUC','logloss','AP']}
        fd=[]
        for f in range(5):
            ii=fold==f;sc=measure(y[ii],p[ii]);sc0=measure(y[ii],reference[ii])
            fd.append({k:sc[k]-sc0[k] for k in ['KS','AUC','logloss']})
        records.append({'name':name,'metrics':m,'delta':delta,'fold_deltas':fd})
    write(root/'scores.json',records)
    # One preregistered candidate, with pair-sampling repetitions averaged
    # inside each bootstrap draw, not counted as independent observations.
    rng=np.random.default_rng(20261020);groups=[np.flatnonzero((fold==f)&(y==cl)) for f in range(5) for cl in [0,1]]
    samples=[];candidates=[pred[f'rank_s{s}'] for s in plan['ranking_seeds']]
    for b in range(plan['bootstrap_repetitions']):
        if b%100==0:check_time(root)
        ii=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);yy=y[ii]
        kb=base.scores(yy,reference[ii]);changes=[]
        for p in candidates:
            m=base.scores(yy,p[ii]);changes.append([m['KS']-kb['KS'],m['AUC']-kb['AUC']])
        samples.append(np.mean(changes,axis=0))
    ci=np.quantile(samples,[.0125,.9875],axis=0) # Bonferroni: two metrics
    chosen=[r for r in records if r['name'].startswith('rank_')]
    stable=all(r['delta']['KS']>0 and r['delta']['AUC']>0 and r['delta']['logloss']<=.0005
        and sum(x['KS']>0 for x in r['fold_deltas'])>=4
        and min(x['KS'] for x in r['fold_deltas'])>=-.003
        and sum(x['AUC']>0 for x in r['fold_deltas'])>=3 for r in chosen)
    passed=bool(stable and ci[0,0]>0 and ci[0,1]>0)
    decision={'candidate_accepted':passed,'selected_model':'ranking' if passed else 'baseline',
        'KS_delta_mean':float(np.mean([r['delta']['KS'] for r in chosen])),
        'AUC_delta_mean':float(np.mean([r['delta']['AUC'] for r in chosen])),
        'KS_adjusted_CI':ci[:,0].tolist(),'AUC_adjusted_CI':ci[:,1].tolist(),
        'fold_and_seed_stability':bool(stable),'public_improvement_guaranteed':False,
        'independent_confirmation':False,'historical_selection_corrected':False,
        'uncertainty':'paired stratified fold/class bootstrap; conditional on fitted predictions; two-metric Bonferroni',
        'point_in_time_certified':False,'plan_sha':base.sha(root/'plan.json')}
    write(root/'decision.json',decision);print(json.dumps(decision),flush=True)
    with (root/'exposures.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'purpose':'terminal fixed-candidate assessment','files':['scores.json','decision.json'],'all_labels_previously_exposed':True})+'\n')
    return decision

def package(root):
    decision=read(root/'decision.json');plan=read(root/'plan.json');p=root/'model';p.mkdir(exist_ok=False)
    train=pd.read_parquet(root/'input/data/train_features.parquet')
    shutil.copy2(root/'input/BASELINE_LOCK.json',p/'baseline_lock.json')
    for f in range(5):
        d=p/f'fold{f}';d.mkdir();source=root/f'input/artifacts/fold{f}'
        selected=(root/f'fits/fold{f}/rank_s{plan["ranking_seeds"][0]}.joblib' if decision['candidate_accepted'] else source/'correction.joblib')
        shutil.copy2(selected,d/'correction.joblib');shutil.copy2(source/'spline.joblib',d/'spline.joblib')
        shutil.copy2(source/'indices.npz',d/'indices.npz')
    lock=read(p/'baseline_lock.json');cols=list(dict.fromkeys(lock['Kumo']['features']+lock['correction']['features']+lock['original_spline']['features']))
    train[['application_id','target']+cols].to_parquet(p/'context_data.parquet',index=False)
    write(p/'schema.json',{'features':cols,'categorical':[c for c in cols if not pd.api.types.is_numeric_dtype(train[c])],
        'target_excluded_from_prediction':True,'input_level':'engineered application feature table',
        'output':'one probability per application in input order'})
    write(p/'manifest.json',{'selected':decision['selected_model'],'ranking_seed':plan['ranking_seeds'][0] if decision['candidate_accepted'] else None,
        'training_policy':'five retained context/correction/spline fold pipelines; all 70000 labels used across the ensemble; no unvalidated 64k-context substitution',
        'context_fit_on_predict':True,'backbone_sha':lock['Kumo']['checkpoint_sha256'],
        'backbone_remote':'/home/taqiyudinmiftah/adit/Logika/insight/goal054_offset_baseline_20261019/checkpoints/kumo_large/classifier.pt',
        'public_score':'0.5647 reported for inherited baseline; no score promised for fresh execution',
        'files':{str(f.relative_to(p)):base.sha(f) for f in p.rglob('*') if f.is_file()}})
    verify_package(p)
    return p

def verify_package(p):
    m=read(p/'manifest.json')
    for name,h in m['files'].items():assert base.sha(p/name)==h,name
    return m

def predict(model,features,output,raw_directory=None,checkpoint=None):
    """Cached raw scores or fresh local-backbone GPU inference; no hidden downloads."""
    manifest=verify_package(model);lock=read(model/'baseline_lock.json');query=pd.read_parquet(features)
    assert 'target' not in query,'Prediction input must not contain labels'
    schema=read(model/'schema.json');assert set(schema['features']).issubset(query.columns)
    assert query.application_id.is_unique and len(query)>0
    predictions=np.zeros(len(query))
    gpu=None
    if raw_directory is None:
        assert checkpoint and base.sha(checkpoint)==manifest['backbone_sha'],'Provide the verified Kumo Large checkpoint'
        import gpu_backend
        gpu=gpu_backend.Engine(checkpoint,lock['Kumo']['features'],model/'context_data.parquet')
    for f in range(5):
        if raw_directory is not None:
            cached=pd.read_parquet(raw_directory/f'fold{f}.parquet')
            assert cached.application_id.is_unique and set(cached.application_id)==set(query.application_id)
            raw=cached.set_index('application_id').prediction.reindex(query.application_id).to_numpy(float)
        else:raw=gpu.predict_fold(f,np.load(model/f'fold{f}/indices.npz')['context'],query)
        assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
        correction=joblib.load(model/f'fold{f}/correction.joblib');spline=joblib.load(model/f'fold{f}/spline.joblib')
        predictions+=(.75*base.corrected_probability(correction,query,raw)[0]+.25*base.predict_spline(spline,query))/5
    assert np.isfinite(predictions).all()
    output=Path(output);assert not output.exists(),'Refusing to overwrite an existing prediction artifact'
    frame=pd.DataFrame({'application_id':query.application_id,'default_probability':predictions})
    if output.suffix=='.csv':frame.to_csv(output,index=False)
    else:frame.to_parquet(output,index=False)
    return frame

def test():
    rng=np.random.default_rng(93);n=100;x=rng.normal(size=(n,8));x[:,0]=1;y=np.r_[np.ones(30),np.zeros(70)]
    off=rng.normal(size=n);pos=rng.choice(np.arange(30),500);neg=rng.choice(np.arange(30,100),500);beta=rng.normal(size=8)*.01
    fun=lambda b:objective(b,x,off,y,pos,neg)
    error=check_grad(lambda b:fun(b)[0],lambda b:fun(b)[1],beta)
    assert error<1e-6,error
    a=objective(beta,x,off,y,pos,neg,alpha=1,penalty=0)
    c=beta.copy();c[0]+=3
    b=objective(c,x,off,y,pos,neg,alpha=1,penalty=0)
    assert abs(a[0]-b[0])<1e-12 and abs(a[1][0])<1e-12
    for p in [np.ones(n),np.round(expit(off),1),expit(off)]:measure(y.astype(int),p)
    print(json.dumps({'test':'gradient, pair shift invariance, tied KS/AUC','gradient_error':error,'pass':True}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['train','evaluate','package','predict','test'])
    parser.add_argument('--root',type=Path);parser.add_argument('--fold',type=int,choices=range(5))
    parser.add_argument('--model',type=Path);parser.add_argument('--features',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--raw-directory',type=Path);parser.add_argument('--checkpoint',type=Path);a=parser.parse_args()
    if a.action=='train':fold_train(a.root,a.fold)
    elif a.action=='evaluate':evaluate(a.root)
    elif a.action=='package':print(package(a.root))
    elif a.action=='predict':predict(a.model,a.features,a.output,a.raw_directory,a.checkpoint)
    else:test()
