"""Fresh frozen-Kumo fitting on identical V25 folds, with one fold resident.

Frozen backbone fitting means in-context adaptation, not gradient fine-tuning.
Outer labels are evaluation-only; the correction learns from the same6k rows.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import argparse
import ast
import ctypes
import gc
import hashlib
import inspect
import importlib.metadata
import json
import resource
import sys
import time
import traceback
import textwrap
import types
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'
os.environ['HF_HUB_CACHE']=str(V/'kumo_gpu/hf_cache')
os.environ['HF_HUB_OFFLINE']='1'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]


def write(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def memory():
    status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
    b=Path('/sys/fs/cgroup');st=dict(line.split() for line in (b/'memory.stat').read_text().splitlines())
    return dict(rss_MiB=int(status['VmRSS'].split()[0])/1024,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                cgroup_current_GiB=int((b/'memory.current').read_text())/2**30,cgroup_limit_GiB=int((b/'memory.max').read_text())/2**30,
                inactive_file_GiB=int(st.get('inactive_file',0))/2**30)


def retain_gpu_context_cache(model):
    """Change only the library's unconditional CPU cache offload, per instance.

    The installed package and other users' model instances remain unchanged.
    Fit preprocessing, sampling, batch order and model execution are preserved.
    """
    original=model.fit.__func__
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source)
    removed=[]
    class KeepGPU(ast.NodeTransformer):
        def visit_If(self,node):
            if ast.unparse(node.test)=='x.is_cuda and len(contexts) > 1':
                assert 'pin_memory=True' in ast.unparse(node) and 'device=\'cpu\'' in ast.unparse(node)
                removed.append(node.lineno)
                return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)
    tree=KeepGPU().visit(tree);ast.fix_missing_locations(tree)
    assert len(removed)==1,'Pinned SDM CPU-offload block changed; review before patching'
    namespace={};exec(compile(tree,'<sdm_fit_gpu_cache_only>','exec'),original.__globals__,namespace)
    model.fit=types.MethodType(namespace['fit'],model)
    return dict(patch='Only skip CPU offload of completed estimator cache batches',original_fit_sha256=hashlib.sha256(source.encode()).hexdigest(),patched_fit_sha256=hashlib.sha256(ast.unparse(tree).encode()).hexdigest(),removed_blocks=len(removed),installed_package_unchanged=True)


def main(name):
    started=time.time();out=V/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists(),'Completed experiment is immutable'
    specs=json.loads((V/'kumo_plan.json').read_text());spec=next(s for s in specs if s['id']==name)
    gate=json.loads((V/'modeling/reports/REPO_REVIEW_GATE.json').read_text())
    assert gate['experiment']==name and gate['status']=='REVIEWED_FOR_NEXT_RUN'
    req=urllib.request.Request('https://api.github.com/repos/TaqiyudinMiftah/Logika/git/ref/heads/main',headers={'User-Agent':'Logika-research'})
    with urllib.request.urlopen(req,timeout=20) as r:current=json.load(r)
    assert current['object']['sha']==gate['reviewed_head'],'New peer commit requires review'
    setup=json.loads((V/'kumo_gpu/SETUP_STATUS.json').read_text());assert setup['status']=='READY',setup
    try:Path('/proc/self/oom_score_adj').write_text('500')
    except OSError:pass
    def state(stage,**kw):
        value=dict(status=stage,experiment=name,pid=os.getpid(),UTC=time.time(),memory=memory(),**kw)
        write(value,out/'STATUS.json');print(json.dumps(value),flush=True)
    state('IMPORTING_GPU_RUNTIME')
    import numpy as np
    import pandas as pd
    import torch
    import sdm
    import joblib
    from baseline import sha,scores,ids_hash,corrected_probability
    from experiments import fit_head,bootstrap
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    weights=json.loads((V/'kumo_gpu/WEIGHTS_MANIFEST.json').read_text())
    classifier=V/'kumo_gpu/hf_cache/models--nvidia--Kumo-Tabular/blobs/cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
    assert classifier.exists() and sha(classifier)==classifier.name
    lock=json.loads((ROOT/'BASELINE_LOCK.json').read_text())
    old=json.loads((V/'runs/V25/RESULT.json').read_text())
    context_columns=lock['Kumo']['features'];head_columns=old['columns']
    assert len(context_columns)==90 and len(head_columns)==18
    cols=list(dict.fromkeys(['application_id','target']+context_columns+head_columns[:12]))
    tr=pd.read_parquet(ROOT/'data/train_features.parquet',columns=cols)
    te=pd.read_parquet(ROOT/'data/test_features.parquet',columns=[c for c in cols if c!='target'])
    # These six raw evidence counts are independent of target and fold fitting.
    support=old['support_features'];additional=old['new_features']
    common=pd.read_parquet(V/'data/features_raw.parquet',columns=['application_id']+support)
    counts=pd.read_parquet(V/'extension/data/raw_features.parquet',columns=['application_id','ext_aux_allocation_negative_n','ext_aux_allocation_nonnegative_n']).rename(columns={'ext_aux_allocation_negative_n':additional[0],'ext_aux_allocation_nonnegative_n':additional[1]})
    common=common.merge(counts,on='application_id',validate='one_to_one')
    tr=tr.merge(common,on='application_id',validate='one_to_one');te=te.merge(common,on='application_id',validate='one_to_one')
    del common,counts;gc.collect()
    mode=spec['context_features']
    assert mode in ['locked90','locked90_plus_v25_support','locked90_plus_v25_velocity','locked90_plus_v25_spectral']
    derived=[];formula={};cache_manifest={}
    if mode!='locked90':context_columns=list(dict.fromkeys(context_columns+support+additional))
    def divide(a,b):
        a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float));out=np.full(a.shape,np.nan)
        np.divide(a,b,out=out,where=np.isfinite(a)&np.isfinite(b)&(b>1e-8));return out
    if mode=='locked90_plus_v25_velocity':
        for d in [tr,te]:
            recent=d.v2_ecom_30d_spend.to_numpy(float);prior=d.v2_ecom_60d_spend.to_numpy(float)-recent
            assert np.nanmin(prior)>=-1e-4,'30/60d spending windows inconsistent'
            d['kumo_spend_30_vs_prior30']=divide(recent,np.maximum(prior,0))
            d['kumo_spend_daily_acceleration']=(recent-prior)/30.
            d['kumo_order_30_vs_prior30']=divide(d.v2_ecom_30d_orders,d.v2_ecom_60d_orders-d.v2_ecom_30d_orders)
            d['kumo_spend30_to_observed_income']=divide(recent,d.v2_account_3m_income_like_credits_mean)
            d['kumo_surplus_after_new_installment']=d.v2_account_3m_income_like_credits_mean-d.v2_account_3m_outflows_mean-d.proposed_monthly_installment
            d['kumo_new_installment_to_min_balance']=divide(d.proposed_monthly_installment,d.v2_account_3m_minimum_balance_min)
        formula=dict(kumo_spend_30_vs_prior30='spend30/(spend60-spend30)',kumo_spend_daily_acceleration='(spend30-(spend60-spend30))/30',kumo_order_30_vs_prior30='orders30/(orders60-orders30)',kumo_spend30_to_observed_income='completed spend30 / mean income_like_credits of last3 complete account months',kumo_surplus_after_new_installment='mean3m income-like credits - mean3m account outflows - proposed new installment; do not add historical/ecommerce expenses again',kumo_new_installment_to_min_balance='new installment / minimum balance of last3 complete account months')
        derived=list(formula)
    elif mode=='locked90_plus_v25_spectral':
        files={'light_fe/data/spectral_features.parquet':['beh_spectral_spend_power_7d','beh_spectral_spend_power_14d','beh_spectral_spend_power_30d','beh_spectral_spend_periodicity_strength','beh_spectral_orders_power_7d','beh_spectral_orders_power_14d','beh_spectral_orders_power_30d','beh_spectral_orders_periodicity_strength','beh_spectral_window_supported']}
        for rel,names in files.items():
            cache=pd.read_parquet(V/rel,columns=['application_id']+names).set_index('application_id');assert cache.index.is_unique
            for d in [tr,te]:
                assert d.application_id.isin(cache.index).all();d[names]=cache.reindex(d.application_id).to_numpy()
            cache_manifest[rel]=sha(V/rel);derived+=names;del cache
        formula={c:('Hann-detrended180-complete-day periodogram band power or concentration/support; exact V80 cache, excluding unstable dominant-cycle argmax' if c.startswith('beh_') else 'Completed item_total category-value entropy, with30d/90d or30-vs-prior60 difference; strict event cutoff; no repeated order_total sum') for c in derived}
    context_columns+=derived
    write(dict(mode=mode,context_columns=context_columns,new_context_inputs=derived,formulas=formula,undefined_ratios='NA whenever denominator <=1e-8 or numerator/denominator nonfinite; never silently zero',event_cache_sha256=cache_manifest,target_used_in_derivation=False),out/'CONTEXT_INPUT_AUDIT.json')
    assert not {'target','customer_id','application_id','decision_datetime'}&set(context_columns)
    sample=pd.read_csv(ROOT/'data/sample_submission.csv');y=tr.target.to_numpy(int)
    ref=pd.read_csv(V/'runs/V25/OOF.csv');assert ref.application_id.tolist()==tr.application_id.tolist()
    reference=ref.prediction.to_numpy();fold_ids=np.full(len(tr),-1);oof=np.full(len(tr),np.nan);pt=np.zeros(len(te));checks=[]
    def table(frame,columns):
        x=frame[columns].copy();stypes={}
        for c in columns:
            if pd.api.types.is_numeric_dtype(x[c]):x[c]=x[c].replace([np.inf,-np.inf],np.nan);stypes[c]='numerical'
            else:x[c]=x[c].astype('string');stypes[c]='categorical'
        return sdm.TableTensor.from_pandas(x,stypes=stypes,device='cuda')
    state('BUILDING_GPU_INPUTS',context_features=len(context_columns))
    x=table(tr,context_columns);xt=table(te,context_columns)
    yt=sdm.TableTensor.from_pandas(pd.DataFrame({'target':y}),stypes={'target':'categorical'},device='cuda')
    features_hash=hashlib.sha256(pd.util.hash_pandas_object(tr[['application_id']+context_columns],index=False).values.tobytes()).hexdigest()
    # Compare against the original SDM implementation before using GPU caches.
    state('VERIFYING_GPU_CACHE_PREDICTION_PARITY')
    idx=np.load(ROOT/'artifacts/fold0/indices.npz');small_ci=idx['context'][:512];small_qi=idx['validation'][:64]
    assert len(np.unique(y[small_ci]))==2 and not np.intersect1d(small_ci,small_qi).size
    check_model=sdm.models.KumoTabular(task='classification',size='large',device='cuda')
    for parameter in check_model.parameters():parameter.requires_grad_(False)
    pilot=[];patch_audit=None
    for keep_gpu in [False,True]:
        if keep_gpu:patch_audit=retain_gpu_context_cache(check_model)
        torch.manual_seed(252026)
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
            check_model.fit(x[small_ci],yt[small_ci],num_estimators=2,estimator_batch_size=1,generator=torch.Generator(device='cuda').manual_seed(252026))
            pilot.append(check_model.predict(x[small_qi])[:, '1'].numerical.reshape(-1).float().cpu().numpy())
    discrepancy=float(np.max(abs(pilot[0]-pilot[1])));assert discrepancy<=1e-6,discrepancy
    patch_audit.update(status='PASS',maximum_absolute_prediction_difference=discrepancy,context_rows=512,query_rows=64,estimators=2,batch_size=1,criterion=1e-6,assertion='Storage optimization only; the cache retains original tensor values and ordering')
    write(patch_audit,out/'GPU_CACHE_PARITY.json');print('GPU_CACHE_PARITY',json.dumps(patch_audit),flush=True)
    del check_model;gc.collect();torch.cuda.empty_cache()
    # Return freed checkpoint/pilot CPU staging buffers to the host allocator.
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    for fold in range(5):
        a=ROOT/'artifacts'/f'fold{fold}';idx=np.load(a/'indices.npz');ci,hi,vi=[idx[k] for k in ['context','correction','validation']]
        assert (len(ci),len(hi),len(vi))==(50000,6000,14000)
        assert len(set(np.r_[ci,hi,vi]))==70000
        fold_ids[vi]=fold
        receipt=out/f'fold{fold}_audit.json'
        if receipt.exists():
            row=json.loads(receipt.read_text());assert row['source_sha256']==sha(__file__)
            saved=pd.read_parquet(out/f'fold{fold}_validation.parquet');test=pd.read_parquet(out/f'fold{fold}_test.parquet')
            assert saved.application_id.tolist()==tr.application_id.iloc[vi].tolist() and test.application_id.tolist()==te.application_id.tolist()
            oof[vi]=saved.prediction;pt+=pd.Series(test.prediction.to_numpy(),index=te.application_id).reindex(sample.application_id).to_numpy()/5
            checks.append(row);continue
        torch.cuda.reset_peak_memory_stats();predictions={};context_audits=[]
        if spec.get('reuse_kumo_from'):
            source=V/'runs'/spec['reuse_kumo_from'];source_result=json.loads((source/'RESULT.json').read_text())
            assert source_result['feature_matrix_sha256']==features_hash and source_result['context_columns']==context_columns
            assert source_result['frozen_Kumo_fraction']==1. and source_result['all_outer_indices_identical']
            for part,ids in [('holdout',tr.application_id.iloc[hi]),('oof',tr.application_id.iloc[vi]),('test',te.application_id)]:
                cached=pd.read_parquet(source/f'fold{fold}_kumo_{part}.parquet');assert cached.application_id.tolist()==ids.tolist()
                predictions[part]=cached.prediction.to_numpy()
            context_audits=[dict(reused_frozen_source=spec['reuse_kumo_from'],reason='Exactly identical Kumo inputs/folds/fit settings; only final coverage gate differs')]
        else:
            bags=spec.get('context_bags',1)
            for bag in range(bags):
                context=ci;seed=2026+fold+bag*10000
                if bags>1:
                    from sklearn.model_selection import train_test_split
                    context,_=train_test_split(ci,train_size=spec['bag_context_rows'],stratify=y[ci],random_state=33031+fold+100*bag)
                assert not np.intersect1d(context,np.r_[hi,vi]).size
                np.save(out/f'fold{fold}_bag{bag}_context_indices.npy',context)
                context_audits.append(dict(bag=bag,rows=len(context),seed=seed,context_ids_sha256=ids_hash(tr,context),event_rate=float(y[context].mean()),eligible_pool_event_rate=float(y[ci].mean()),disjoint_from_correction_and_validation=True))
                state('LOADING_KUMO_WEIGHTS',fold=fold,bag=bag);torch.manual_seed(seed)
                model=sdm.models.KumoTabular(task='classification',size='large',device='cuda');retain_gpu_context_cache(model)
                for parameter in model.parameters():parameter.requires_grad_(False)
                assert all(not p.requires_grad for p in model.parameters())
                state('FITTING_KUMO_CONTEXT',fold=fold,bag=bag,context_rows=len(context),estimators=spec['estimators'],GPU_allocated_GiB=torch.cuda.memory_allocated()/2**30)
                with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
                    model.fit(x[context],yt[context],num_estimators=spec['estimators'],estimator_batch_size=spec['estimator_batch_size'],generator=torch.Generator(device='cuda').manual_seed(seed))
                qbatch=spec['query_batch_size']
                for part,q,ids in [('holdout',x[hi],tr.application_id.iloc[hi]),('oof',x[vi],tr.application_id.iloc[vi]),('test',xt,te.application_id)]:
                    pred=[]
                    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
                        for j in range(0,len(q),qbatch):
                            state('PREDICTING_KUMO',fold=fold,bag=bag,partition=part,rows_complete=j,rows_total=len(q))
                            p=model.predict(q[j:j+qbatch])[:, '1'].numerical.reshape(-1).float().cpu().numpy();pred.append(p)
                    p=np.concatenate(pred);assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                    predictions[part]=predictions.get(part,np.zeros(len(p)))+p/bags
                del model;gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)
        for part,ids in [('holdout',tr.application_id.iloc[hi]),('oof',tr.application_id.iloc[vi]),('test',te.application_id)]:
            pd.DataFrame(dict(application_id=ids.to_numpy(),prediction=predictions[part])).to_parquet(out/f'fold{fold}_kumo_{part}.parquet',index=False)
        peak_gpu=torch.cuda.max_memory_allocated()/2**30
        state('FITTING_OFFSET_HEAD',fold=fold,backbone_peak_gpu_GiB=peak_gpu)
        h,val,test=tr.iloc[hi],tr.iloc[vi],te
        for part,d in [('correction',h),('validation',val),('test',test)]:
            cached=pd.read_parquet(V/'extension/data'/f'fold{fold}'/(part+'.parquet'),columns=['application_id']+support+additional)
            assert cached.application_id.tolist()==d.application_id.tolist()
            for c in support+additional:np.testing.assert_equal(d[c].to_numpy(),cached[c].to_numpy())
        head=fit_head(h,predictions['holdout'],head_columns,spec['penalty'],spec['strength'])
        ph,dh=corrected_probability(head,h,predictions['holdout']);pv,dv=corrected_probability(head,val,predictions['oof']);tp,dt=corrected_probability(head,test,predictions['test'])
        sv=pd.read_parquet(a/'spline_oof.parquet');st=pd.read_parquet(a/'spline_test.parquet')
        assert sv.application_id.tolist()==val.application_id.tolist() and st.application_id.tolist()==te.application_id.tolist()
        p=.75*pv+.25*sv.prediction.to_numpy();tp=.75*tp+.25*st.prediction.to_numpy()
        if spec.get('coverage_fallback'):
            assert spec['coverage_fallback']=='V25_if_allocation_count_zero'
            for part,d,pred in [('validation',val,p),('test',test,tp)]:
                anchor=pd.read_parquet(V/'runs/V25'/f'fold{fold}_{part}.parquet');assert anchor.application_id.tolist()==d.application_id.tolist()
                observed=d.aux_allocation_count.to_numpy(float)>0
                pred[~observed]=anchor.prediction.to_numpy()[~observed]
        oof[vi]=p;pt+=pd.Series(tp,index=te.application_id).reindex(sample.application_id).to_numpy()/5
        raw_control=pd.read_parquet(a/'kumo_oof.parquet');assert raw_control.application_id.tolist()==val.application_id.tolist()
        row=dict(fold=fold,metrics=scores(y[vi],p),baseline=scores(y[vi],reference[vi]),delta_KS=scores(y[vi],p)['KS']-scores(y[vi],reference[vi])['KS'],delta_AUC=scores(y[vi],p)['AUC']-scores(y[vi],reference[vi])['AUC'],head_training_KS=scores(y[hi],ph)['KS'],heldout_corrected_Kumo_KS=scores(y[vi],pv)['KS'],train_validation_head_KS_gap=scores(y[hi],ph)['KS']-scores(y[vi],pv)['KS'],parameters=len(head['coefficients']),correction_rows=6000,
                 context_ids_sha256=ids_hash(tr,ci),train_ids_sha256=ids_hash(tr,hi),validation_ids_sha256=ids_hash(tr,vi),source_sha256=sha(__file__),backbone_max_GPU_GiB=peak_gpu,memory=memory(),frozen_parameters_fraction=1.,old_raw_Kumo_mean_abs_error=float(np.mean(abs(predictions['oof']-raw_control.prediction.to_numpy()))),old_raw_Kumo_max_abs_error=float(np.max(abs(predictions['oof']-raw_control.prediction.to_numpy()))))
        row['context_fit_audit']=context_audits
        joblib.dump(dict(head=head,kumo_weight=.75,blend='probability'),out/f'fold{fold}_correction.joblib',compress=3)
        pd.DataFrame(dict(application_id=val.application_id,prediction=p,correction_logit=dv)).to_parquet(out/f'fold{fold}_validation.parquet',index=False)
        pd.DataFrame(dict(application_id=te.application_id,prediction=tp)).to_parquet(out/f'fold{fold}_test.parquet',index=False)
        write(dict(frozen_weights=str(classifier),weights_sha256=classifier.name,revision=weights['revision'],context_columns=context_columns,context_ids_file=str(a/'indices.npz'),seed=2026+fold,estimators=spec['estimators'],estimator_batch_size=spec['estimator_batch_size'],correction_checkpoint=str(out/f'fold{fold}_correction.joblib')),out/f'fold{fold}_checkpoint.json')
        write(dict(context_fits=context_audits,coverage_fallback=spec.get('coverage_fallback'),feature_matrix_sha256=features_hash),out/f'fold{fold}_context_audit.json')
        write(row,receipt);checks.append(row);state('FOLD_COMPLETE',fold=fold,metrics=row['metrics'],delta_KS=row['delta_KS'])
        gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
    del x,xt,yt;gc.collect();torch.cuda.empty_cache()
    assert np.isfinite(oof).all() and np.isfinite(pt).all()
    submission=V/'submissions'/f'submission_{name}_Kumo_Large_V25.csv'
    pd.DataFrame(dict(application_id=sample.application_id,default_probability=pt)).to_csv(submission,index=False)
    pd.DataFrame(dict(application_id=tr.application_id,target=y,fold=fold_ids,prediction=oof,baseline_prediction=reference)).to_csv(out/'OOF.csv',index=False)
    ci=bootstrap(y,reference,oof,fold_ids);m,b=scores(y,oof),scores(y,reference)
    from modeling_features import error_analysis
    frames=[]
    for fold in range(5):
        ids=np.load(ROOT/'artifacts'/f'fold{fold}'/'indices.npz')['validation']
        frames.append(dict(vi=ids,val=tr.iloc[ids]))
    error_analysis(out,tr,y,fold_ids,reference,oof,frames)
    method='Frozen Kumo Large;50k labeled context,8 estimators; disjoint6k offset-head training;14k outer validation; .75*sigmoid(logit(p_Kumo)+.5*Xbeta)+.25*p_original_spline77; mean5fold test probabilities'
    if spec.get('context_bags'):method=method.replace('50k labeled context',f"mean{spec['context_bags']} stratified contexts of{spec['bag_context_rows']} from eligible50k pool")
    if spec.get('coverage_fallback'):method+='; final=V25 for allocation_count<=0/missing, otherwise spectral expert. Gate fixed from exploratory prior OOF error analysis; selection bias remains.'
    result=dict(status='COMPLETE',experiment=name,spec=spec,columns=head_columns,context_columns=context_columns,new_features=additional,support_features=support,OOF=m,baseline=b,delta_KS=m['KS']-b['KS'],delta_AUC=m['AUC']-b['AUC'],comparison_baseline='V25',mean_fold_KS=float(np.mean([r['metrics']['KS'] for r in checks])),std_fold_KS=float(np.std([r['metrics']['KS'] for r in checks],ddof=1)),delta_mean_fold_KS=float(np.mean([r['delta_KS'] for r in checks])),fold_wins=sum(r['delta_KS']>0 for r in checks),folds=checks,confidence_interval=ci,submission=str(submission.relative_to(ROOT)),submission_sha256=sha(submission),public_score=None,elapsed_seconds=time.time()-started,source_sha256=sha(__file__),modeling_formula=method,modeling_objective='Unweighted BCE + .003/2*sum(beta[1:]**2); full correction fitting and .5 shrink at inference',controller_source='v7/scripts/kumo_gpu_experiment.py',frozen_Kumo_fraction=1.,backbone_retrained=True,backbone_gradient_updated=False,in_context_refitted=True,original_spline_unchanged=True,all_outer_indices_identical=True,point_in_time_certified=False,inherited_baseline_snapshot_uncertainty=True,features_changed_vs_V25=spec['context_features']!='locked90',memory=memory(),GPU_cache_parity=patch_audit,weights_manifest=weights,feature_matrix_sha256=features_hash,packages={k:importlib.metadata.version(k) for k in ['torch','structured-data-models','numpy','pandas','scikit-learn']},overfitting_assessment=dict(development_folds_reused=True,independent_holdout=False,hyperparameters_prespecified=True,retrained_seeds=1,selection_corrected_CI=False,public_score_unknown=True,risk='HIGH_SEARCH_MULTIPLICITY and new CUDA runtime; matched fresh control required'),error_analysis=dict(json=str((out/'ERROR_ANALYSIS.json').relative_to(ROOT)),markdown=str((out/'ERROR_ANALYSIS.md').relative_to(ROOT)),cases=str((out/'ERROR_CASES.csv').relative_to(ROOT))))
    result['derived_context_features']=derived;result['derived_context_formulas']=formula
    result['context_event_cache_sha256']=cache_manifest
    if name!='V168':
        control=pd.read_csv(V/'runs/V168/OOF.csv');assert control.application_id.tolist()==tr.application_id.tolist()
        result['fresh_control_delta_KS']=m['KS']-scores(y,control.prediction)['KS']
        result['fresh_control_confidence_interval']=bootstrap(y,control.prediction.to_numpy(),oof,fold_ids)
    result['controller_source']='v7/scripts/kumo_gpu_experiment_v2.py'
    write(result,out/'RESULT.json');write(dict(method=method,spec=spec,context_columns=context_columns,head_columns=head_columns,derived_context_formulas=formula,weights=weights,source_sha256=sha(__file__)),out/'MODELING_METHOD.json')
    write(dict(control='V25',candidate_metrics=m,control_metrics=b,delta_KS=result['delta_KS'],delta_AUC=result['delta_AUC'],confidence_interval=ci,head_gap_candidate=float(np.mean([r['train_validation_head_KS_gap'] for r in checks])),head_gap_control=float(np.mean([r['train_validation_head_KS_gap'] for r in old['folds']]))),out/'MATCHED_CONTROL.json')
    (out/'REPORT.md').write_text(f"# {name}: {spec['label']}\n\n{method}\n\nOOF KS={m['KS']:.8f}; AUC={m['AUC']:.8f}; delta KS against cached V25={result['delta_KS']:+.8f}.95% paired delta CI={ci['delta_KS']}.\n\nAll5folds evaluated. Frozen weights have no gradient updates. In-context fitting was rerun. Torch/CUDA runtime differs from archived control; V168 is a fresh runtime control. Development search bias remains, and a public score is unknown. V25 remains the public incumbent.\n\nError analysis is held-out only. Legacy snapshot provenance remains unverified.\n")
    state('COMPLETE',metrics=m,delta_KS=result['delta_KS'],target_reached=m['KS']>=.531)
    from master_log import generate
    generate()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    try:main(args.run)
    except Exception:
        out=V/'runs'/args.run
        write(dict(status='FAILED',pid=os.getpid(),traceback=traceback.format_exc(),memory=memory()),out/'STATUS.json')
        raise
