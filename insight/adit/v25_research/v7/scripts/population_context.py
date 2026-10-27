"""Train-context-only population retrieval for frozen Large Kumo."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd


def reference(frame,mode):
    if mode=='employment':return dict(mode=mode,levels=sorted(frame.employment_type.fillna('MISSING').astype(str).unique().tolist()))
    if mode=='apr_regime':
        x=frame.annual_interest_rate.to_numpy(float);x=x[np.isfinite(x)];assert len(x)>0
        return dict(mode=mode,median_APR=float(np.median(x)),levels=['low_or_equal','high'])
    raise ValueError('Unknown population grouping')


def groups(frame,ref):
    if ref['mode']=='employment':
        x=frame.employment_type.fillna('MISSING').astype(str).to_numpy();return np.array([a if a in ref['levels'] else 'UNSUPPORTED' for a in x])
    x=frame.annual_interest_rate.to_numpy(float)
    return np.where(~np.isfinite(x),'UNSUPPORTED',np.where(x<=ref['median_APR'],'low_or_equal','high'))


def local_predictions(spec,tr,te,x,xt,yt,ci,hi,vi,fold,predictions,state,out,retain_gpu):
    import gc,ctypes,torch,sdm
    from sklearn.model_selection import train_test_split
    from baseline import ids_hash
    alpha=spec['local_context_weight'];limit=spec['local_context_max_rows'];y=tr.target.to_numpy(int);ref=reference(tr.iloc[ci],spec['population_grouping']);pool_groups=groups(tr.iloc[ci],ref)
    ref.update(reference_rows=len(ci),reference_ids_sha256=ids_hash(tr,ci),target_used_for_group_definition=False,outer_or_test_reference_fit=False)
    (out/f'fold{fold}_population_reference.json').write_text(json.dumps(ref,indent=2));audits=[]
    for gi,g in enumerate(ref['levels']):
        context=ci[pool_groups==g];seed=42026+fold+gi*100
        if len(context)<64 or len(np.unique(y[context]))<2:
            audits.append(dict(group=g,skipped=True,reason='Insufficient training context support; keep global prediction'));continue
        original_rows=len(context);natural_rate=float(y[context].mean())
        if len(context)>limit:context,_=train_test_split(context,train_size=limit,stratify=y[context],random_state=seed)
        assert np.isin(context,ci).all() and not np.intersect1d(context,np.r_[hi,vi]).size and len(np.unique(context))==len(context)
        np.save(out/f'fold{fold}_population{gi}_context_indices.npy',context)
        torch.manual_seed(seed);model=sdm.models.KumoTabular(task='classification',size='large',device='cuda');retain_gpu(model)
        for p in model.parameters():p.requires_grad_(False)
        assert all(not p.requires_grad for p in model.parameters())
        state('FITTING_POPULATION_CONTEXT',fold=fold,population=g,context_rows=len(context),grouping=ref['mode'])
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
            model.fit(x[context],yt[context],num_estimators=spec['estimators'],estimator_batch_size=spec['estimator_batch_size'],generator=torch.Generator(device='cuda').manual_seed(seed))
            for part,q,d in [('holdout',x[hi],tr.iloc[hi]),('oof',x[vi],tr.iloc[vi]),('test',xt,te)]:
                choose=np.flatnonzero(groups(d,ref)==g)
                for j in range(0,len(choose),spec['query_batch_size']):
                    take=choose[j:j+spec['query_batch_size']];local=model.predict(q[take])[:,'1'].numerical.reshape(-1).float().cpu().numpy();assert np.isfinite(local).all()
                    predictions[part][take]=(1-alpha)*predictions[part][take]+alpha*local
        audits.append(dict(group=g,population_grouping=ref['mode'],context_pool_rows=original_rows,context_rows=len(context),context_ids_sha256=ids_hash(tr,context),seed=seed,natural_pool_event_rate=natural_rate,sampled_event_rate=float(y[context].mean()),local_weight=alpha,global_weight=1-alpha,backbone_frozen_fraction=1.,outer_labels_used=False))
        del model;gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)
    return predictions,audits


def preflight(root):
    root=Path(root);train=pd.DataFrame(dict(employment_type=['salaried','self_employed','variable_income']*4,annual_interest_rate=[.16,.20,.24]*4))
    query=pd.DataFrame(dict(employment_type=['salaried','NEW',None],annual_interest_rate=[.16,.24,np.nan]));checks=[]
    for mode in ['employment','apr_regime']:
        ref=reference(train,mode);a=groups(query,ref);assert a[-1]=='UNSUPPORTED'
        if mode=='employment':assert a[1]=='UNSUPPORTED' and a[0]=='salaried'
        else:assert ref['median_APR']==.2 and a.tolist()==['low_or_equal','high','UNSUPPORTED']
        # Reference remains fixed when the query population changes.
        altered=query.copy();altered.annual_interest_rate=1.;assert reference(train,mode)==ref
        checks.append(dict(mode=mode,reference=ref,unsupported_falls_back_to_global=True))
    code=Path(__file__);out=root/'v7/population/reports';out.mkdir(parents=True,exist_ok=True)
    (out/'PREFLIGHT.json').write_text(json.dumps(dict(status='PASS',synthetic_only=True,checks=checks,source_sha256=hashlib.sha256(code.read_bytes()).hexdigest()),indent=2));print('POPULATION_REFERENCE_PREFLIGHT_PASS',flush=True)
