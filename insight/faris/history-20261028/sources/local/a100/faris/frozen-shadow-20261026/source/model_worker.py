"""Model-only experiments on sealed features. Retain recipe/predictions, no weights."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('SDM_CHUNK_MEMORY_FRACTION','0.02')
os.environ['HF_HUB_OFFLINE']='1'
import argparse,gc,importlib,json,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.special import expit,logit
from common import R,sha,dump,utc,ks

def numeric(x,context):
    out=[]
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]):
            v=x[c].to_numpy(float);v=np.sign(v)*np.log1p(abs(v));median=np.nanmedian(v[context]);median=0 if not np.isfinite(median) else median
            missing=~np.isfinite(v);v=np.where(missing,median,v);sd=max(np.std(v[context]),1e-4);mean=np.mean(v[context])
            out.extend([np.clip((v-mean)/sd,-8,8),missing.astype(float)])
        else:
            s=x[c].astype('string').fillna('__MISSING__')
            for category in sorted(s.iloc[context].unique()):out.append((s==category).to_numpy(float))
    z=np.column_stack(out).astype('float32');assert np.isfinite(z).all();return z
def make_kumo(size,seed,device='cuda'):
    import sdm,torch
    checkpoint=Path(os.environ['HF_HOME'])/f'hub/models--nvidia--Kumo-Tabular/snapshots/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/{size}/classifier.pt'
    want=json.loads((R/'CHECKPOINTS.json').read_text())[size];assert sha(checkpoint)==want
    module=importlib.import_module(sdm.models.KumoTabular.__module__)
    def resolve(**kw):
        assert kw['repo_id']=='nvidia/Kumo-Tabular' and kw['filename']==f'{size}/classifier.pt' and kw['revision']=='v1.0.1'
        return str(checkpoint)
    with patch.object(module,'download_checkpoint',resolve):return sdm.models.KumoTabular(task='classification',size=size,device=device)
def kumo_fit(x,trainrows,y,queryrows,spec,seed):
    import torch,sdm
    torch.set_num_threads(2);torch.set_grad_enabled(False);device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.85)
    ordered=x.iloc[np.r_[trainrows,queryrows]].reset_index(drop=True).copy();n=len(trainrows)
    types={}
    for c in ordered:
        if pd.api.types.is_numeric_dtype(ordered[c]):types[c]='numerical'
        else:
            vocabulary=set(ordered[c].iloc[:n].dropna().astype(str));ordered[c]=ordered[c].astype('string').where(ordered[c].astype('string').isin(vocabulary));types[c]='categorical'
    table=sdm.TableTensor.from_pandas(ordered,stypes=types,device=device);target=sdm.TableTensor.from_pandas(pd.DataFrame({'target':y}),stypes={'target':'categorical'},device=device)
    model=make_kumo(spec['size'],seed,device);pred=[];stage_start=time.time()
    with torch.inference_mode(),torch.amp.autocast(device,dtype=torch.float16 if device=='cuda' else torch.bfloat16,enabled=device=='cuda'):
        model.fit(table[:n],target,num_estimators=spec['estimators'],estimator_batch_size=1,generator=torch.Generator(device=device).manual_seed(seed))
        print(json.dumps({'stage':'kumo_context_fitted','device':device,'context':n,'seconds':time.time()-stage_start}),flush=True)
        for start in range(n,len(table),512):
            pred.append(model.predict(table[start:start+512])[:,'1'].numerical.reshape(-1).float().cpu().numpy())
            if len(pred)==1 or len(pred)%10==0:print(json.dumps({'stage':'kumo_predicting','device':device,'query_done':min(len(pred)*512,len(queryrows)),'query_total':len(queryrows),'seconds':time.time()-stage_start}),flush=True)
        length=min(512,len(queryrows));again=model.predict(table[n:n+length])[:,'1'].numerical.reshape(-1).float().cpu().numpy()
    raw=np.concatenate(pred);assert np.max(abs(raw[:length]-again))<1e-7
    model.clear();del table,target,model;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return raw
def contextual(x,roles,spec,seed):
    from sklearn.cluster import MiniBatchKMeans
    ctx=roles['context'];query=np.r_[roles['correction'],roles['validation']]
    routing=['loan_amount','annual_interest_rate','installment_observed_income_ratio','bureau_score_mean','repayment_historical_dpd_mean','v2_account_3m_average_balance_mean']
    z=numeric(x[routing],ctx);km=MiniBatchKMeans(n_clusters=spec['clusters'],random_state=seed,n_init=3,batch_size=2048).fit(z[ctx]);cluster=km.predict(z[query])
    rng=np.random.default_rng(seed);result=np.full(len(query),np.nan);Y=roles['y_context']
    for k in range(spec['clusters']):
        selected=np.flatnonzero(cluster==k)
        if not len(selected):continue
        distance=((z[ctx]-km.cluster_centers_[k])**2).sum(1)
        nearest=np.argsort(distance,kind='stable')[:spec['local_rows']]
        remainder=np.setdiff1d(np.arange(len(ctx)),nearest);global_part=rng.choice(remainder,min(2000,len(remainder)),replace=False)
        members=np.r_[nearest,global_part];assert np.unique(Y[members]).size==2
        result[selected]=kumo_fit(x,ctx[members],Y[members],query[selected],spec,seed+k)
    assert np.isfinite(result).all();return result
def neural(x,roles,spec,seed,baseline):
    import torch
    from torch import nn
    torch.manual_seed(seed);torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.5)
    ctx=roles['context'];query=np.r_[roles['correction'],roles['validation']];z=numeric(x,ctx);v=torch.tensor(z,device=device)
    rank,choice=train_test_split(np.arange(6000),test_size=.5,stratify=roles['y_correction'],random_state=261026)
    ranking=spec['kind']=='ranking'
    tr=roles['correction'][rank] if ranking else ctx
    y=roles['y_correction'][rank] if ranking else roles['y_context']
    y=torch.tensor(y,dtype=torch.float32,device=device);net=nn.Sequential(nn.Linear(z.shape[1],spec['width']),nn.SiLU(),nn.Dropout(.1),nn.Linear(spec['width'],64),nn.SiLU(),nn.Linear(64,1)).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=spec['lr'],weight_decay=spec['decay']);rng=np.random.default_rng(seed)
    teacher=torch.tensor(logit(np.clip(baseline['raw'][:6000][rank],1e-5,1-1e-5)),dtype=torch.float32,device=device) if ranking else None
    for epoch in range(spec['epochs']):
        net.train();order=rng.permutation(len(tr))
        for start in range(0,len(tr),512):
            ids=order[start:start+512];offset=teacher[ids] if ranking else 0
            correction=.5*torch.tanh(net(v[tr[ids]]).squeeze(1)) if ranking else net(v[tr[ids]]).squeeze(1)
            logits=offset+correction;loss=nn.functional.binary_cross_entropy_with_logits(logits,y[ids])
            if ranking:
                positive=torch.nonzero(y[ids]>0,as_tuple=True)[0];negative=torch.nonzero(y[ids]==0,as_tuple=True)[0]
                if len(positive) and len(negative):
                    a=positive[torch.randint(len(positive),(1024,),device=device)];b=negative[torch.randint(len(negative),(1024,),device=device)]
                    loss=loss+spec['pair_weight']*nn.functional.softplus(-(logits[a]-logits[b])).mean()
                loss=loss+spec['consistency']*correction.square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
    net.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(query),1024):out.append(net(v[query[start:start+1024]]).squeeze(1).cpu().numpy())
    p=np.concatenate(out)
    return expit(logit(np.clip(baseline['raw'],1e-5,1-1e-5))+.5*np.tanh(p)) if ranking else expit(p)
def run(taskid,fold):
    start=time.time();spec=json.loads((R/'CONFIGS.json').read_text())[taskid];dest=R/f'runs/{taskid}/fold{fold}';dest.mkdir(parents=True,exist_ok=False)
    try:
        seal=json.loads((R/'FEATURE_FREEZE.json').read_text());manifest=json.loads((R/f'frozen/fold{fold}/MANIFEST.json').read_text())
        p=R/f'frozen/fold{fold}/features.parquet';assert sha(p)==manifest['features_sha256']
        x=pd.read_parquet(p)[manifest['views'][spec['view']]];roles=np.load(R/f'frozen/fold{fold}/roles.npz');baseline=np.load(R/f'frozen/fold{fold}/baseline.npz')
        ctx=roles['context'];query=np.r_[roles['correction'],roles['validation']];seed=spec['seed']+fold
        dump({'status':'running','config':spec,'fold':fold,'seed':seed,'columns':list(x),'context_rows':len(ctx),
              'feature_sha256':sha(p),'freeze_sha256':sha(R/'FEATURE_FREEZE.json'),'source_sha256':sha(__file__),
              'outer_labels_loaded':False,'host':__import__('socket').gethostname(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'Kumo_precision':'float32' if os.environ.get('CUDA_VISIBLE_DEVICES')=='' else 'autocast_float16',
              'generator_backend':'cpu' if os.environ.get('CUDA_VISIBLE_DEVICES')=='' else 'cuda','UTC':utc()},dest/'START.json')
        kind=spec['kind']
        if kind=='kumo':raw=kumo_fit(x,ctx,roles['y_context'],query,spec,seed)
        elif kind=='contextual':raw=contextual(x,roles,spec,seed)
        elif kind in ['mlp','ranking']:raw=neural(x,roles,spec,seed,baseline)
        elif kind in ['lgb','cat','xgb','extra','logistic']:
            rank,_=train_test_split(np.arange(6000),test_size=.5,stratify=roles['y_correction'],random_state=261026)
            inner=roles['correction'][rank]
            if kind=='cat':
                from catboost import CatBoostClassifier
                x=x.copy();cats=x.select_dtypes(include=['object','string','category']).columns.tolist()
                for c in cats:x[c]=x[c].fillna('__MISSING__').astype(str)
                model=CatBoostClassifier(**spec['params'],random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
                model.fit(x.iloc[ctx],roles['y_context'],cat_features=cats,eval_set=(x.iloc[inner],roles['y_correction'][rank]),early_stopping_rounds=150)
                raw=model.predict_proba(x.iloc[query])[:,1]
            else:
                z=numeric(x,ctx)
                if kind=='lgb':
                    import lightgbm as lgb
                    model=lgb.LGBMClassifier(**spec['params'],random_state=seed,n_jobs=2,verbosity=-1)
                    model.fit(z[ctx],roles['y_context'],eval_set=[(z[inner],roles['y_correction'][rank])],callbacks=[lgb.early_stopping(150,verbose=False)])
                elif kind=='xgb':
                    from xgboost import XGBClassifier
                    model=XGBClassifier(**spec['params'],random_state=seed,n_jobs=2,early_stopping_rounds=150)
                    model.fit(z[ctx],roles['y_context'],eval_set=[(z[inner],roles['y_correction'][rank])],verbose=False)
                elif kind=='extra':
                    from sklearn.ensemble import ExtraTreesClassifier
                    model=ExtraTreesClassifier(**spec['params'],random_state=seed,n_jobs=2).fit(z[ctx],roles['y_context'])
                else:
                    from sklearn.linear_model import LogisticRegression
                    model=LogisticRegression(**spec['params'],max_iter=1000).fit(z[ctx],roles['y_context'])
                raw=model.predict_proba(z[query])[:,1]
        else:raise ValueError(kind)
        raw=np.asarray(raw,dtype=float);assert len(raw)==20000 and np.isfinite(raw).all() and 0<=raw.min()<=raw.max()<=1
        np.savez_compressed(dest/'prediction.npz',rows=query,raw=raw)
        dump({'status':'complete','seconds':time.time()-start,'prediction_sha256':sha(dest/'prediction.npz'),
              'inner_ranking_KS':ks(roles['y_correction'],raw[:6000]),'UTC':utc(),'fitted_weights_saved':False,
              'max_RSS_KiB':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
              'execution_device':'cpu' if os.environ.get('CUDA_VISIBLE_DEVICES')=='' else 'cuda'},dest/'RESULT.json')
        print(json.dumps({'task':taskid,'fold':fold,'status':'complete','seconds':time.time()-start}),flush=True)
    except BaseException:dump({'status':'failed','error':traceback.format_exc(),'seconds':time.time()-start},dest/'FAILURE.json');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--fold',type=int,required=True);a=p.parse_args();run(a.config,a.fold)
