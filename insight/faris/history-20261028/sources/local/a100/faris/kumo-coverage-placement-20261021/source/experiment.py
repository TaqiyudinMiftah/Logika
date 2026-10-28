"""Public-supported coverage information in Kumo, with two matched controls."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,time,gc,hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
import feature_head
import spline as fixed_spline
from gpu_backend import Engine

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['control90','coverage96','null96']
EXTRA=['risk_conditional_negative_months','risk_conditional_nonnegative_months',
       'aux_account_months','aux_orders90','aux_ecommerce_observed_days','aux_allocation_count']
def read(path):return json.loads(Path(path).read_text())
def write(path,value):b.dump(value,path)
def emit(stage,**values):print(json.dumps(dict(time=time.time(),stage=stage,**values)),flush=True)
def check():assert time.time()<read(ROOT/'plan.json')['original_deadline'] and not (ROOT/'STOP').exists()
def load(fold=None):
    plan=read(ROOT/'plan.json');inp=Path(plan['input']);parent=Path(plan['feature_parent'])
    assert read(parent/'INDEPENDENT_AUDIT.json')['status']=='passed'
    frame=pd.read_parquet(inp/'data/train_features.parquet')
    counts=pd.read_parquet(parent/'data/features.parquet')
    assert len(frame)==70000 and frame.application_id.is_unique and frame.customer_id.is_unique
    assert frame.application_id.tolist()==counts.application_id.tolist()
    for col in EXTRA:
        assert col not in frame;frame[col]=counts[col].to_numpy(float)
    lock=read(inp/'BASELINE_LOCK.json')
    assert len(lock['Kumo']['features'])==90 and not set(EXTRA)&set(lock['Kumo']['features'])
    if fold is not None:
        repaired=Path(plan['repaired_features'])/f'fold{fold}'
        complete=read(repaired/'complete.json')
        assert complete['training_only_cohort_statistics'] and complete['no_target_labels_available_to_feature_estimators']
        assert b.sha(repaired/'features.parquet')==complete['feature_table_sha256']
        fixed=pd.read_parquet(repaired/'features.parquet').set_index('application_id').loc[frame.application_id]
        assert fixed.customer_id.tolist()==frame.customer_id.tolist()
        required=set(lock['Kumo']['features']+lock['correction']['features']+lock['original_spline']['features'])
        assert required<=set(fixed.columns)
        for name in required:frame[name]=fixed[name].to_numpy()
        ix=dict(np.load(inp/f'artifacts/fold{fold}/indices.npz'))
        assert b.ids_hash(frame,ix['context'])==complete['fit_ids_sha256']
    return plan,inp,frame,lock
def indices(inp,frame,f):
    d=inp/f'artifacts/fold{f}';ix=dict(np.load(d/'indices.npz'));meta=read(d/'metadata.json')
    assert len(ix['context'])==50000 and len(ix['correction'])==6000 and len(ix['validation'])==14000
    assert not np.intersect1d(ix['context'],ix['correction']).size
    assert not np.intersect1d(ix['train'],ix['validation']).size
    assert np.array_equal(np.sort(np.r_[ix['context'],ix['correction']]),ix['train'])
    for name,label in [('context','context'),('correction','correction_training'),('validation','validation')]:
        assert b.ids_hash(frame,ix[name])==meta[label+'_ids_sha256']
    return ix
def tables(frame,ix,variant,f):
    context=frame.iloc[ix['context']].reset_index(drop=True).copy()
    rows=np.r_[ix['correction'],ix['validation']]
    query=frame.iloc[rows].drop(columns='target').reset_index(drop=True).copy()
    permutations={}
    if variant=='null96':
        rng=np.random.default_rng(20261101+f)
        permutations['context']=rng.permutation(len(context))
        permutations['correction']=rng.permutation(6000)
        permutations['validation']=rng.permutation(14000)+6000
        context[EXTRA]=context[EXTRA].to_numpy()[permutations['context']]
        query[EXTRA]=query[EXTRA].to_numpy()[np.r_[permutations['correction'],permutations['validation']]]
        assert not np.intersect1d(permutations['correction'],permutations['validation']).size
    return context,query,rows,permutations
def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    assert np.isfinite(p).all() and np.all((p>0)&(p<1))
    out=b.scores(y,p);out['logloss']=float(log_loss(y,p))
    assert abs(out['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-12
    n1=y.sum();auc=float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1)))
    assert abs(out['AUC']-auc)<1e-12
    return out
def preflight():
    check();plan,inp,data,lock=load()
    assert b.sha(plan['checkpoint'])==plan['checkpoint_sha']
    seen=np.zeros(len(data),int)
    for f in range(5):
        ix=indices(inp,data,f);seen[ix['validation']]+=1
        _,_,fold_data,_=load(f)
        context,query,rows,perm=tables(fold_data,ix,'null96',f)
        assert np.array_equal(np.sort(perm['context']),np.arange(50000))
        assert np.array_equal(np.sort(perm['correction']),np.arange(6000))
        assert np.array_equal(np.sort(perm['validation']),np.arange(6000,20000))
        # Null shuffling preserves six-column vectors and never uses labels.
        assert np.array_equal(context[EXTRA].to_numpy(),fold_data.iloc[ix['context']][EXTRA].to_numpy()[perm['context']],equal_nan=True)
        assert not set(EXTRA)&{'target','application_id','customer_id'}
    assert np.all(seen==1)
    # Recheck original V25 heads on these newly merged raw coverage columns.
    ix=indices(inp,data,0);valid=data.iloc[ix['validation']].drop(columns='target')
    original=joblib.load(Path(plan['feature_parent'])/'evidence/runs/V25/fold0_correction.joblib')
    assert original['columns']==lock['correction']['features']+EXTRA
    raw=pd.read_parquet(inp/'artifacts/fold0/kumo_oof.parquet').prediction.to_numpy()
    spline=pd.read_parquet(inp/'artifacts/fold0/spline_oof.parquet').prediction.to_numpy()
    saved=pd.read_parquet(Path(plan['feature_parent'])/'evidence/runs/V25/fold0_validation.parquet').prediction.to_numpy()
    replay=.75*b.corrected_probability(original,valid,raw)[0]+.25*spline
    assert np.max(abs(replay-saved))<1e-12
    # Device admission occurs before torch/CUDA initialization.
    import subprocess
    while True:
        check()
        listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        ready=[int(a) for a,c in [line.split(',') for line in listing.splitlines()] if float(c)>=11000]
        if ready:break
        time.sleep(10)
    os.environ['CUDA_VISIBLE_DEVICES']=str(ready[0])
    smoke=ROOT/'smoke';smoke.mkdir(exist_ok=True)
    assert not (smoke/'fitted_context.pt').exists(),'Preserve prior preflight artifacts'
    _,_,fold_data,_=load(0)
    context=fold_data.iloc[ix['context'][:512]].copy();query=fold_data.iloc[ix['correction'][:64]].drop(columns='target').copy()
    engine=Engine(plan['checkpoint'],lock['Kumo']['features']+EXTRA,context)
    pred=engine.fit_predict(np.arange(len(context)),query,2026,smoke/'fitted_context.pt')
    # Independently read the serialized bytes once to validate the streaming hash.
    assert b.sha(smoke/'fitted_context.pt')==engine.last_cache['sha256']
    repeated=engine.replay(smoke/'fitted_context.pt',query)
    error=float(np.max(abs(pred-repeated)));assert error<2e-5,error
    assert all(not p.requires_grad for p in engine.model.parameters())
    import torch
    gpu=dict(device=torch.cuda.get_device_name(),physical_device=ready[0],peak_vram=torch.cuda.max_memory_allocated())
    np.savez_compressed(smoke/'predictions.npz',prediction=pred,replayed=repeated)
    write(smoke/'cache.json',engine.last_cache)
    write(ROOT/'preflight.json',dict(status='passed',streaming_hash_independently_verified=True,
        actual_cache_replay_max_error=error,actual_predictions=len(pred),accelerator=gpu,
        V25_head_replay_error=float(np.max(abs(replay-saved))),full_OOF_partition_rows=70000,
        null_shuffle_partition_isolation=True,new_default_labels_used_for_raw_features=False,
        repaired_prior_fit_scope='Each frozen50kcontext only; query covariates do not estimate shared statistics.',
        historical_V25_replay_is_not_the_matched_repaired_control=True))
    emit('preflight_pass',gpu=gpu,cache_replay_error=error)
def spline(f):
    check();plan,inp,data,lock=load(f);ix=indices(inp,data,f)
    dest=ROOT/f'shared/fold{f}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists(),'Preserve completed spline'
    training=data.iloc[ix['train']].copy();query=data.iloc[ix['validation']].drop(columns='target').copy()
    assert len(training)==56000 and not set(training.application_id)&set(query.application_id)
    model=fixed_spline.fit(training,lock['original_spline']['features'])
    prediction=b.predict_spline(model,query)
    joblib.dump(model,dest/'spline.joblib',compress=3)
    np.savez_compressed(dest/'validation.npz',rows=ix['validation'],prediction=prediction)
    write(dest/'boundary.json',dict(training_ids_sha=model['fit_ids_sha'],validation_ids_sha=b.ids_hash(data,ix['validation']),
        fit_rows=56000,validation_labels_passed_to_fit=False,prior_fit_rows=50000,
        original_spline_predictions_reused=False,parameters=model['resolved_parameters']))
    write(dest/'complete.json',dict(status='complete',files={p.name:b.sha(p)for p in dest.iterdir()if p.is_file()}))
    emit('spline_complete',fold=f)

def condition(f,variant):
    check();assert variant in VARIANTS
    assert read(ROOT/'SCIENTIFIC_GATE.json')['status']=='passed_for_bounded_development_study'
    plan,inp,data,lock=load(f);ix=indices(inp,data,f)
    dest=ROOT/f'conditions/fold{f}/{variant}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists(),'Do not overwrite completed condition'
    context,query,rows,permutations=tables(data,ix,variant,f)
    columns=lock['Kumo']['features']+([] if variant=='control90' else EXTRA)
    context[['application_id','target']+columns].to_parquet(dest/'context.parquet',index=False)
    query[['application_id']+columns].to_parquet(dest/'query.parquet',index=False)
    np.savez_compressed(dest/'row_indices.npz',context=ix['context'],correction=ix['correction'],validation=ix['validation'])
    if permutations:np.savez_compressed(dest/'null_permutations.npz',**permutations)
    emit('Kumo_fit_started',fold=f,variant=variant,context_rows=len(context),query_rows=len(query),features=len(columns),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    engine=Engine(plan['checkpoint'],columns,context)
    raw=engine.fit_predict(np.arange(50000),query,2026+f,dest/'fitted_context.pt')
    assert len(raw)==20000 and np.isfinite(raw).all()
    write(dest/'cache.json',engine.last_cache)
    del engine;gc.collect()
    np.savez_compressed(dest/'raw_predictions.npz',rows=rows,prediction=raw)
    # The head always receives true own coverage, including in the null condition.
    train=data.iloc[ix['correction']].copy();valid=data.iloc[ix['validation']].drop(columns='target').copy()
    head=feature_head.fit(train,raw[:6000],lock['correction']['features']+EXTRA,.003,.5)
    joblib.dump(head,dest/'correction.joblib',compress=3)
    p=b.corrected_probability(head,valid,raw[6000:])[0]
    shared=ROOT/f'shared/fold{f}';sm=read(shared/'complete.json')
    for name,digest in sm['files'].items():assert b.sha(shared/name)==digest
    sp=np.load(shared/'validation.npz');assert np.array_equal(sp['rows'],ix['validation'])
    final=.75*p+.25*sp['prediction']
    np.savez_compressed(dest/'validation.npz',rows=ix['validation'],prediction=final)
    emit('head_complete',fold=f,variant=variant,parameters=len(head['coefficients']))
    write(dest/'boundary.json',dict(context_ids_sha=b.ids_hash(data,ix['context']),
        correction_ids_sha=b.ids_hash(data,ix['correction']),validation_ids_sha=b.ids_hash(data,ix['validation']),
        feature_columns=columns,head_columns=head['columns'],seed=2026+f,backbone_frozen=True,
        query_default_labels_passed_to_Kumo=False,head_fit_rows=6000,spline_fit_rows=56000,
        original_deadline=plan['original_deadline'],cohort_prior_fit_rows=50000,
        original_pooled_prior_spline_reused=False))
    small={p.name:b.sha(p) for p in dest.iterdir() if p.is_file() and p.name not in ['fitted_context.pt','complete.json']}
    write(dest/'complete.json',dict(status='complete',small_files=small,
        large_cache=read(dest/'cache.json'),large_cache_hash_method='SHA256 of the exact bytes during serialization; independent read audit follows training without gating other workers'))
    emit('condition_complete',fold=f,variant=variant)
def finish():
    check();plan,inp,data,lock=load();y=data.target.to_numpy(int)
    preds={v:np.full(len(data),np.nan) for v in VARIANTS+['published_V25']};foldids=np.full(len(data),-1)
    replay_error=0.;hashes=0
    for f in range(5):
        ix=indices(inp,data,f);foldids[ix['validation']]=f
        _,_,fold_data,_=load(f)
        for v in VARIANTS:
            d=ROOT/f'conditions/fold{f}/{v}';meta=read(d/'complete.json')
            for name,digest in meta['small_files'].items():assert b.sha(d/name)==digest;hashes+=1
            assert (d/'fitted_context.pt').stat().st_size==meta['large_cache']['bytes']
            z=np.load(d/'validation.npz');assert np.array_equal(z['rows'],ix['validation'])
            raw=np.load(d/'raw_predictions.npz');assert np.array_equal(raw['rows'],np.r_[ix['correction'],ix['validation']])
            model=joblib.load(d/'correction.joblib')
            sp=np.load(ROOT/f'shared/fold{f}/validation.npz')['prediction']
            repeated=.75*b.corrected_probability(model,fold_data.iloc[ix['validation']].drop(columns='target'),raw['prediction'][6000:])[0]+.25*sp
            error=float(np.max(abs(repeated-z['prediction'])));assert error<1e-12;replay_error=max(replay_error,error)
            preds[v][ix['validation']]=z['prediction']
        ref=pd.read_parquet(Path(plan['feature_parent'])/f'evidence/runs/V25/fold{f}_validation.parquet')
        assert ref.application_id.tolist()==data.application_id.iloc[ix['validation']].tolist()
        preds['published_V25'][ix['validation']]=ref.prediction.to_numpy()
    scores={name:metrics(y,p) for name,p in preds.items()}
    folds={name:[metrics(y[foldids==f],p[foldids==f]) for f in range(5)] for name,p in preds.items()}
    assert abs(scores['published_V25']['KS']-.5302846600440411)<1e-12
    comparison={name:dict(delta={k:scores['coverage96'][k]-scores[name][k] for k in ['KS','AUC','logloss']},
        KS_fold_deltas=[a['KS']-c['KS'] for a,c in zip(folds['coverage96'],folds[name])])
        for name in ['control90','null96','published_V25']}
    for c in comparison.values():c['KS_fold_wins']=int(np.sum(np.array(c['KS_fold_deltas'])>0))
    # Predeclared signal-vs-control and signal-vs-null paired intervals.
    structures={}
    for name in VARIANTS:
        p=preds[name];order=np.argsort(p,kind='stable');structures[name]=(order,np.r_[p[order][1:]!=p[order][:-1],True])
    groups=[np.flatnonzero((foldids==f)&(y==c)) for f in range(5) for c in [0,1]]
    rng=np.random.default_rng(20261102);draws={name:[] for name in ['control90','null96']}
    for rep in range(2000):
        if rep%200==0:check();emit('bootstrap',completed=rep)
        w=np.zeros(len(y),int)
        for group in groups:w[group]=rng.multinomial(len(group),np.full(len(group),1/len(group)))
        n1=w[y==1].sum();n0=w[y==0].sum();m={}
        for name,(order,end) in structures.items():
            cp=np.cumsum(w[order]*y[order])[end];cn=np.cumsum(w[order]*(1-y[order]))[end]
            dp=np.diff(np.r_[0,cp]);dn=np.diff(np.r_[0,cn])
            m[name]=np.array([np.max(abs(cp/n1-cn/n0)),np.sum(dp*(cn-.5*dn))/(n1*n0)])
        for name in draws:draws[name].append(m['coverage96']-m[name])
    for name,dd in draws.items():comparison[name]['adjusted_conditional_interval']=np.quantile(dd,[.00625,.99375],axis=0).tolist()
    np.savez_compressed(ROOT/'oof.npz',target=y,fold=foldids,**preds)
    np.savez_compressed(ROOT/'bootstrap.npz',**{name:np.asarray(dd) for name,dd in draws.items()})
    result=dict(status='training_and_scoring_complete',metrics=scores,fold_metrics=folds,comparisons=comparison,
        primary='coverage96',point_goal_reached=bool(scores['coverage96']['KS']>=.54),
        fitted_Kumo_contexts=15,fitted_correction_heads=15,fitted_spline_models=5,OOF_rows=70000,small_artifact_hashes_verified=hashes,
        max_correction_replay_error=replay_error,cache_integrity='Streaming SHA256 retained; independent full reads pending separate cache audit.',
        interval_columns=['KS','AUC'],interval_scope='2000 paired fold/class bootstrap, Bonferroni two controls/two metrics; conditional on fitted predictions, excludes historical search and retraining uncertainty',
        AUC_policy='Complementary evidence, reported without an automatic zero-decline veto.',
        public_score_reference=.56822,public_score_source='User report for exact V25 CSV; coverage96 not submitted',
        historical_label_reuse=True,point_in_time_certified=False,submission_generated=False,
        all_shared_priors_fit_inside_context=True,shared_spline_refitted_inside_outer_training=True,
        historical_V25_comparison_is_not_single_change=True)
    write(ROOT/'result.json',result)
    lines=['# Kumo coverage placement','','Three fixed input conditions; five complete folds each.','',
           '| Condition | OOF KS | AUC | Log loss |','|---|---:|---:|---:|']
    for name,m in scores.items():lines.append(f'| {name} | {m["KS"]:.8f} | {m["AUC"]:.8f} | {m["logloss"]:.8f} |')
    lines+=['',f'Primary .54 KS point target reached: {result["point_goal_reached"]}.',
        'All contexts, heads, preprocessing, inputs and predictions retained. Independent cache-file audit follows separately.',
        'Original labels and public feedback have been reused; conditional intervals do not prove new private generalization.',
        'No new competition submission was created.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');emit('training_and_scoring_complete',primary=scores['coverage96'])
def audit_caches():
    """One full read per retained cache; does not gate admissions of other jobs."""
    checks=[]
    for f in range(5):
        for v in VARIANTS:
            check();d=ROOT/f'conditions/fold{f}/{v}';meta=read(d/'cache.json')
            digest=b.sha(d/'fitted_context.pt');assert digest==meta['sha256'],(f,v)
            checks.append(dict(fold=f,variant=v,sha256=digest,bytes=meta['bytes']))
            write(ROOT/'CACHE_AUDIT_PROGRESS.json',dict(verified=len(checks),total=15,checks=checks))
            emit('retained_cache_verified',fold=f,variant=v,completed=len(checks))
    write(ROOT/'CACHE_AUDIT.json',dict(status='passed',independent_cache_hashes=15,
        total_bytes=sum(c['bytes'] for c in checks),checks=checks,
        full_GPU_cache_replay_scope='Actual saved-cache reload exercised in preflight; full-size contexts independently hashed, not all rerun.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','spline','condition','finish','audit_caches'])
    p.add_argument('--fold',type=int);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    if a.action=='preflight':preflight()
    elif a.action=='spline':spline(a.fold)
    elif a.action=='condition':condition(a.fold,a.variant)
    elif a.action=='finish':finish()
    else:audit_caches()
