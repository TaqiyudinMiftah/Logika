"""Five additional hypotheses for frozen Kumo; fold-local references only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def sources(spec):
    mode=spec['context_features']
    if mode.endswith('_bayes'):return ['aux_allocation_late_share']
    if mode.endswith('_provider'):return ['aux_Bureau_A','aux_Bureau_B','aux_Bureau_C','aux_bureau_freshness']
    return []

def prepare(spec,tr,te,V):
    mode=spec['context_features'];formula={}
    if mode.endswith('_bayes'):
        formula=dict(kumo_allocation_bayes_mean='(alpha+k)/(alpha+beta+n)',kumo_allocation_bayes_variance='(alpha+k)*(beta+n-k)/((alpha+beta+n)^2*(alpha+beta+n+1))',kumo_allocation_credibility='n/(n+alpha+beta)',kumo_allocation_bayes_burden='posterior mean * proposed-installment/observed-income; undefined affordability remains NA')
        for d in [tr,te]:
            for c in formula:d[c]=np.nan
    elif mode.endswith('_latent'):
        definition=json.loads((V.parent/'provenance/definition.json').read_text())
        names=['re_price_posterior_sd','re_bureau_posterior_sd','re_declared_income_log_gap','re_outflow_income_geometric_ratio']
        assert all(c in definition['columns'] and c in tr for c in names)
        formula={c:'Existing locked latent measurement supplied directly to Kumo; same value/provenance as baseline correction; inherited global-reference uncertainty remains' for c in names}
    elif mode.endswith('_basket'):
        names=['light_basket_value_entropy_30d','light_basket_value_entropy_90d','light_basket_value_entropy_change_30_vs_prior60','light_basket_completed_item_rows_30d','light_basket_observed_categories_30d']
        cache=pd.read_parquet(V/'light_fe/data/basket_value_features.parquet',columns=['application_id']+names).set_index('application_id');assert cache.index.is_unique
        for d in [tr,te]:
            assert d.application_id.isin(cache.index).all();d[names]=cache.reindex(d.application_id).to_numpy()
            categories=d.light_basket_observed_categories_30d.to_numpy(float);entropy=d.light_basket_value_entropy_30d.to_numpy(float)
            value=np.full(len(d),np.nan);one=(categories==1)&np.isfinite(entropy);value[one]=0.
            many=(categories>1)&np.isfinite(entropy);value[many]=entropy[many]/np.log(categories[many]);d['kumo_basket_normalized_entropy']=value
        formula={c:'Completed item_total spending entropy across categories;30d,90d or recent30-vs-prior60; strict event cutoff' for c in names[:3]}
        formula.update(kumo_basket_normalized_entropy='category-value entropy30d/log(observed categories30d);0 for exactly1 observed category, missing for0')
    elif mode.endswith('_provider'):
        formula=dict(kumo_bureau_provider_consensus='mean(provider-specific z scores), with context-only mean/std',kumo_bureau_provider_disagreement='population std of provider z scores, missing if fewer than2 providers',kumo_bureau_provider_coverage='number of observed provider scores',kumo_bureau_freshness='existing mean report-age measure in days',kumo_bureau_stale_disagreement='provider-normalized disagreement * log1p(max(report-age,0))')
        for d in [tr,te]:
            for c in formula:d[c]=np.nan
    elif mode.endswith(('_local_context','_score_anchor')):pass
    else:raise ValueError(mode)
    return list(formula),formula

def fit_fold(spec,tr,te,ci,out,fold):
    mode=spec['context_features'];audit=dict(fold=fold,reference_rows=len(ci),labels_used=False,reference='Only this outer-fold50k Kumo context; no correction/validation/test reference fitting')
    if mode.endswith('_bayes'):
        from extension_features import beta_prior
        def counts(d):
            n=d.aux_allocation_count.to_numpy(float);rate=d.aux_allocation_late_share.to_numpy(float)
            assert np.isfinite(n).all() and (n>=0).all() and np.isfinite(rate[n>0]).all()
            raw=np.where(n>0,n*np.nan_to_num(rate),0.);k=np.rint(raw)
            assert np.max(abs(k-raw))<1e-5 and np.all((k>=0)&(k<=n))
            return k,n
        k,n=counts(tr.iloc[ci]);prior=beta_prior(k,n);a,b=prior['alpha'],prior['beta'];audit['prior']=prior
        for d in [tr,te]:
            k,n=counts(d);aa=a+k;bb=b+n-k;total=aa+bb
            d['kumo_allocation_bayes_mean']=aa/total;d['kumo_allocation_bayes_variance']=aa*bb/(total*total*(total+1))
            d['kumo_allocation_credibility']=n/(n+a+b)
            d['kumo_allocation_bayes_burden']=d.kumo_allocation_bayes_mean*d.installment_observed_income_ratio
        audit['n0_rule']='Posterior equals fitted population prior when n0; explicit n/credibility distinguishes absence of evidence. Allocation delay is not DPD or full-settlement time.'
    elif mode.endswith('_provider'):
        names=['aux_Bureau_A','aux_Bureau_B','aux_Bureau_C'];parameters={}
        for c in names:
            x=tr.iloc[ci][c].dropna().to_numpy(float);assert len(x)>1
            parameters[c]=dict(mean=float(x.mean()),sd=max(float(x.std()),1.))
        audit['provider_reference']=parameters
        for d in [tr,te]:
            z=np.column_stack([(d[c].to_numpy(float)-parameters[c]['mean'])/parameters[c]['sd'] for c in names]);n=np.isfinite(z).sum(1)
            mean=np.divide(np.nansum(z,axis=1),n,out=np.full(len(d),np.nan),where=n>0)
            variance=np.divide(np.nansum((z-mean[:,None])**2,axis=1),n,out=np.full(len(d),np.nan),where=n>=2)
            sd=np.sqrt(variance);age=d.aux_bureau_freshness.to_numpy(float)
            d['kumo_bureau_provider_consensus']=mean;d['kumo_bureau_provider_disagreement']=sd;d['kumo_bureau_provider_coverage']=n
            d['kumo_bureau_freshness']=age;d['kumo_bureau_stale_disagreement']=sd*np.log1p(np.maximum(age,0))
        audit['interpretation']='Standardized measurement proxies, not an identified latent PD or independent debt-account totals'
    else:return None
    (out/f'fold{fold}_feature_reference.json').write_text(json.dumps(audit,indent=2));return audit

def local_predictions(spec,tr,te,x,xt,yt,ci,hi,vi,fold,predictions,state,out,retain_gpu):
    import gc,ctypes,torch,sdm
    from sklearn.model_selection import train_test_split
    from baseline import ids_hash
    alpha=spec['local_context_weight'];limit=spec['local_context_max_rows'];audits=[]
    group=lambda d:np.where(d.aux_allocation_count.to_numpy(float)>0,1,0)
    pool_groups=group(tr.iloc[ci]);y=tr.target.to_numpy(int)
    for g in [0,1]:
        context=ci[pool_groups==g];seed=42026+fold+g*100
        if len(context)<64 or len(np.unique(y[context]))<2:
            audits.append(dict(group=g,skipped=True,reason='Insufficient context or class support; retain global'))
            continue
        if len(context)>limit:context,_=train_test_split(context,train_size=limit,stratify=y[context],random_state=seed)
        assert not np.intersect1d(context,np.r_[hi,vi]).size
        np.save(out/f'fold{fold}_local{g}_context_indices.npy',context)
        torch.manual_seed(seed);model=sdm.models.KumoTabular(task='classification',size='large',device='cuda');retain_gpu(model)
        for p in model.parameters():p.requires_grad_(False)
        assert all(not p.requires_grad for p in model.parameters())
        state('FITTING_LOCAL_CONTEXT',fold=fold,coverage_group=g,context_rows=len(context))
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
            model.fit(x[context],yt[context],num_estimators=spec['estimators'],estimator_batch_size=spec['estimator_batch_size'],generator=torch.Generator(device='cuda').manual_seed(seed))
            for part,q,d in [('holdout',x[hi],tr.iloc[hi]),('oof',x[vi],tr.iloc[vi]),('test',xt,te)]:
                choose=np.flatnonzero(group(d)==g)
                for j in range(0,len(choose),spec['query_batch_size']):
                    take=choose[j:j+spec['query_batch_size']];local=model.predict(q[take])[:, '1'].numerical.reshape(-1).float().cpu().numpy()
                    predictions[part][take]=(1-alpha)*predictions[part][take]+alpha*local
        audits.append(dict(group=g,context_rows=len(context),context_ids_sha256=ids_hash(tr,context),seed=seed,local_weight=alpha,global_weight=1-alpha,backbone_frozen_fraction=1.,outer_labels_used=False))
        del model;gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)
    return predictions,audits
