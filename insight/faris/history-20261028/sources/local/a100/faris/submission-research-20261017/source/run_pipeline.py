"""Frozen research recipe -> terminal assessment -> all-label refit -> submission.csv.

No competition service calls. Existing experiment controllers are never resumed.
"""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse,contextlib,concurrent.futures,hashlib,json,multiprocessing,sys,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from metaexp.features import Bank
from metaexp.pipeline import fit_predict,predict
from metaexp.common import read,write,sha,identity,metrics,json_safe


def tie_ks(y,p):
    y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=float)
    order=np.argsort(p,kind='stable');yy=y[order];v=p[order];last=np.r_[v[1:]!=v[:-1],True]
    delta=np.cumsum(yy)/np.sum(yy)-np.cumsum(1-yy)/np.sum(1-yy)
    return float(np.max(np.abs(delta[last])))

def combine(columns):
    x=np.column_stack(columns);assert x.shape[1]>=3 and np.isfinite(x).all()
    return np.mean(np.sort(x,axis=1)[:,1:-1],axis=1)

def component_fit(arguments):
    bankroot,folder,recipe,mode,train_csv,seed=arguments
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);bank=Bank(Path(bankroot).parent)
    if mode=='confirmation':fitrows=bank.dev;y=bank.y;query=np.load(bank.root/'confirmation_rows.npy')
    else:
        train=pd.read_csv(train_csv,usecols=['application_id','target']).set_index('application_id')
        fitrows=np.flatnonzero(bank.metadata.split.to_numpy()=='train');query=np.flatnonzero(bank.metadata.split.to_numpy()=='test')
        y=train.target.reindex(bank.metadata.application_id.iloc[fitrows]).to_numpy(dtype=np.int64)
        assert len(fitrows)==70000 and len(query)==30000 and set(np.unique(y))=={0,1}
        assert np.array_equal(train.target.reindex(bank.metadata.application_id.iloc[bank.dev]).to_numpy(),bank.y)
    request={'recipe':recipe,'mode':mode,'seed':seed,'fit_rows_sha256':identity(fitrows.tolist()),'query_rows_sha256':identity(query.tolist()),'bank_id':bank.manifest['bank_id']}
    if (folder/'complete.json').exists():
        assert read(folder/'request.json')==request
        for name,h in read(folder/'complete.json')['files'].items():assert sha(folder/name)==h
        return str(folder/'prediction.npy')
    write(folder/'request.json',request);started=time.time()
    with (folder/'training.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        p,pre,m,weight,cdf,messages=fit_predict(bank,recipe,fitrows,y,query,'cpu',seed)
        assert p.shape==(len(query),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        np.save(folder/'prediction.npy',p);np.save(folder/'query_rows.npy',query)
        write(folder/'preprocessing.json',pre.audit);write(folder/'parameters.json',json_safe(m.get_params()))
        # The memory-mapped input bank is reattached explicitly for inference.
        pre.bank=None;joblib.dump({'preprocessing':pre,'model':m,'weight':weight,'recipe':recipe,'seed':seed},folder/'pipeline.joblib',compress=1)
        write(folder/'resources.json',{'elapsed_seconds':time.time()-started,'host':os.uname().nodename,'pid':os.getpid(),'warnings':messages,'train_rows':len(fitrows),'query_rows':len(query)})
    write(folder/'complete.json',{'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name not in ['complete.json','training.log']}})
    return str(folder/'prediction.npy')

def fit_phase(args,recipes,mode,tag='primary'):
    folders=[args.output/mode/tag/k for k in recipes]
    tasks=[(str(args.bank),str(folder),recipes[k],mode,str(args.train),args.seed) for k,folder in zip(recipes,folders)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),mp_context=multiprocessing.get_context('spawn')) as ex:
        outputs=list(ex.map(component_fit,tasks))
    values=[np.load(p) for p in outputs];p=combine(values);dest=args.output/mode/tag
    np.save(dest/'combined.npy',p);write(dest/'components.json',{'order':list(recipes),'aggregation':'discard row maximum and minimum; average remaining probabilities'})
    return p,folders

def assess(args,p,recipe):
    # Prediction artifact is durable before reserved labels are opened.
    assert (args.output/'confirmation/primary/combined.npy').exists()
    y=np.load(args.bank/'confirmation_y.npy').astype(np.int64)
    baseline=np.load(args.evidence/'reference_confirmation.npy');fallback=np.load(args.evidence/'fallback_confirmation.npy')
    assert len(p)==len(y)==len(baseline)==14000
    from scipy.stats import ks_2samp
    score=metrics(y,p);ind=float(ks_2samp(p[y==0],p[y==1]).statistic);assert abs(score['ks']-ind)<1e-12
    delta=score['ks']-tie_ks(y,baseline);rng=np.random.default_rng(20261017);classes=[np.flatnonzero(y==c) for c in [0,1]];d=[]
    for _ in range(500):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in classes]);d.append(tie_ks(y[ix],p[ix])-tie_ks(y[ix],baseline[ix]))
    selected='fallback' if delta<-.01 else 'primary'
    result={'selected':selected,'primary':score,'reference_v3_ks':tie_ks(y,baseline),'primary_delta_ks':delta,'paired_bootstrap_95ci':np.quantile(d,[.025,.975]).tolist(),'bootstrap_samples':500,'fallback_ks':tie_ks(y,fallback),'rule':recipe['holdout_fallback_rule'],'caveat':recipe['caveat'],'recipe_sha256':sha(args.recipe)}
    write(args.output/'confirmation_report.json',result);print(json.dumps({'confirmation':result}),flush=True)
    return result

def validate_submission(csv,sample,expected=None):
    df=pd.read_csv(csv);s=pd.read_csv(sample)
    assert df.columns.tolist()==['application_id','default_probability'] and s.columns.tolist()==df.columns.tolist()
    assert len(df)==30000 and df.application_id.is_unique and df.application_id.tolist()==s.application_id.tolist()
    p=df.default_probability.to_numpy();assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.std(p)>.001
    if expected is not None:assert np.allclose(p,expected,rtol=0,atol=5e-13)
    return {'rows':len(df),'columns':df.columns.tolist(),'sha256':sha(csv),'min_probability':float(p.min()),'max_probability':float(p.max()),'mean_probability':float(p.mean()),'std_probability':float(p.std()),'sample_order_exact':True,'unique_application_ids':True}

def execute(args):
    args.output.mkdir(parents=True,exist_ok=True)
    assert sha(args.recipe)==args.recipe.with_suffix('.sha256').read_text().strip()
    recipe=read(args.recipe);assert recipe['combination']=={'features':1},'This pipeline is frozen to the selected feature ensemble'
    bank=Bank(args.bank.parent);assert bank.manifest['bank_id']==recipe['components']['bank_id']
    assert sha(args.train)==bank.manifest['raw_sha256']['train_applications']
    for name,h in bank.manifest['files'].items():assert sha(args.bank/name)==h,'Feature bank integrity failed: '+name
    source=Path(__file__).parent;manifest={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(args.output/'source_manifest.json',manifest)
    import importlib.metadata
    write(args.output/'environment.json',{'python':sys.version,'host':os.uname().nodename,'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib']}})
    parents=recipe['components']['ensemble']['parents'];recipes={k:recipe['components']['components'][k]['recipe'] for k in parents}
    print(json.dumps({'phase':'confirmation','models':list(recipes),'recipe_sha256':sha(args.recipe)}),flush=True)
    p,_=fit_phase(args,recipes,'confirmation');assessment=assess(args,p,recipe)
    if assessment['selected']=='fallback':
        recipes=recipe['fallback']['components'];print('Applying frozen fallback rule',flush=True)
    print(json.dumps({'phase':'full_refit','models':list(recipes),'labeled_applications':70000}),flush=True)
    p,folders=fit_phase(args,recipes,'full',assessment['selected'])
    query=np.flatnonzero(bank.metadata.split.to_numpy()=='test');sample=pd.read_csv(args.sample)
    values=pd.Series(p,index=bank.metadata.application_id.iloc[query].to_numpy());ordered=sample.application_id.map(values).to_numpy()
    df=pd.DataFrame({'application_id':sample.application_id,'default_probability':ordered});csv=args.output/'submission.csv';tmp=csv.with_suffix('.csv.tmp');df.to_csv(tmp,index=False,float_format='%.12f');tmp.replace(csv)
    # Reload all fitted pipelines and independently replay 256 query rows.
    check=[]
    for folder in folders:
        obj=joblib.load(folder/'pipeline.joblib');obj['preprocessing'].bank=bank
        q=predict(obj['model'],obj['preprocessing'].transform(query[:256]),obj['weight']);saved=np.load(folder/'prediction.npy')[:256]
        assert np.allclose(q,saved,rtol=0,atol=1e-12),'Reloaded model prediction differs'
        check.append(q)
    assert np.allclose(combine(check),p[:256],rtol=0,atol=1e-12)
    result=validate_submission(csv,args.sample,ordered);result.update(recipe_sha256=sha(args.recipe),selected=assessment['selected'],model_reload_rows=256,models=list(recipes),confirmation=assessment)
    write(args.output/'submission_validation.json',result);write(args.output/'complete.json',{'submission_sha256':result['sha256'],'models':{str(f.relative_to(args.output)):sha(f/'pipeline.joblib') for f in folders}})
    print(json.dumps({'state':'completed','submission':str(csv),'validation':result}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',type=Path,required=True);p.add_argument('--train',type=Path,required=True);p.add_argument('--sample',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=5);p.add_argument('--seed',type=int,default=2026);a=p.parse_args()
    for n in ['bank','train','sample','recipe','evidence','output']:setattr(a,n,getattr(a,n).resolve())
    execute(a)
