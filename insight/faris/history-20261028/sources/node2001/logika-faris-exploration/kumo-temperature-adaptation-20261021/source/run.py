"""One preregistered temperature-adaptation recipe plus a matched frozen repeat.

Only context labels reach adaptation. Correction targets are separate; outer
validation labels remain with the PC2 evaluator. No performance-based pruning.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['KUMO_MEMORY_FRACTION']='.30'
from pathlib import Path
import json,time,hashlib,gc,subprocess
import numpy as np
import pandas as pd
import joblib
from gpu_backend import Engine
from preflight import choose_parameters,pretrained_hash,sha
import feature_head
import baseline as b
ROOT=Path(__file__).resolve().parents[1]
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def emit(stage,**values):print(json.dumps(dict(time=time.time(),stage=stage,**values)),flush=True)
def check(plan):
    assert time.time()<plan['original_deadline'],'Original deadline expired'
    assert not (ROOT/'STOP').exists(),'Campaign stopped'
def save_head(fold,variant,raw,query,labels,indices,spline,columns):
    d=ROOT/f'runs/fold{fold}/{variant}';training=query.iloc[:6000].copy();training['target']=labels
    assert len(labels)==6000 and 'target'not in query
    bundle=feature_head.fit(training,raw[:6000],columns,.003,.5)
    joblib.dump(bundle,d/'correction.joblib',compress=3)
    p=b.corrected_probability(bundle,query.iloc[6000:],raw[6000:])[0]
    final=.75*p+.25*spline['prediction']
    assert np.isfinite(final).all()and((final>=0)&(final<=1)).all()
    np.savez_compressed(d/'raw_predictions.npz',rows=np.r_[indices['correction'],indices['validation']],prediction=raw)
    np.savez_compressed(d/'validation.npz',rows=indices['validation'],application_id=query.application_id.iloc[6000:].to_numpy(str),prediction=final)
    write(d/'head_boundary.json',dict(fit_rows=6000,query_rows=14000,feature_columns=columns,
        validation_labels_available=False,head_labels_excluded_from_adaptation=True))
def run_fold(fold,plan):
    check(plan);root=ROOT/f'runs/fold{fold}'
    if (root/'complete.json').exists():
        meta=read(root/'complete.json')
        for name,digest in meta['files'].items():assert sha(root/name)==digest
        emit('retained_fold_reused',fold=fold);return
    assert not root.exists(),'Partial fold preserved; review before explicit retry'
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    assert free>=plan['minimum_free_VRAM_MiB'],f'Insufficient safe GPU headroom:{free}'
    root.mkdir(parents=True);start=time.time();data=ROOT/f'full_data/fold{fold}'
    manifest=read(data/'manifest.json')
    for name,digest in manifest['files'].items():assert sha(data/name)==digest
    context=pd.read_parquet(data/'context.parquet');query=pd.read_parquet(data/'query.parquet')
    labels=np.load(data/'correction_labels.npy');indices=dict(np.load(data/'indices.npz'))
    spline=dict(np.load(data/'spline_validation.npz'));columns=read(ROOT/'data/columns.json')
    head_columns=read(ROOT/'full_data/head_columns.json')
    assert len(context)==50000 and len(query)==20000 and len(labels)==6000
    assert context.application_id.is_unique and query.application_id.is_unique
    assert not set(context.application_id)&set(query.application_id)
    assert 'target'in context and 'target'not in query
    assert np.array_equal(spline['rows'],indices['validation'])
    engine=Engine(plan['checkpoint'],columns,context);t=engine.torch;t.cuda.reset_peak_memory_stats()
    for variant in ['frozen','temperature']:(root/variant).mkdir()
    emit('frozen_control_fit',fold=fold,context_rows=50000,query_rows=20000)
    frozen=engine.fit_predict(np.arange(50000),query,2026+fold,root/'frozen/fitted_context.pt')
    write(root/'frozen/cache.json',engine.last_cache)
    save_head(fold,'frozen',frozen,query,labels,indices,spline,head_columns)
    selected=choose_parameters(engine.model);assert sum(p.numel()for p in selected.values())==384
    original={name:p.detach().clone()for name,p in selected.items()}
    original_hash=pretrained_hash(engine.model,selected)
    optimizer=t.optim.AdamW(list(selected.values()),lr=.001,weight_decay=0)
    rng=np.random.default_rng(20261108+fold);training=plan['proposed_training'];history=[]
    for step in range(training['episodes']):
        check(plan);rows=rng.choice(50000,512,replace=False);support,qi=rows[:384],rows[384:]
        assert not set(support)&set(qi)
        engine.model.train()
        with t.amp.autocast('cuda',dtype=t.float16):
            out=engine.model(engine.x[support],engine.y[support],engine.x[qi],
                num_estimators=1,estimator_batch_size=1,
                generator=t.Generator(device='cuda').manual_seed(2026+fold+1009*step))
            probability=out[:,'1'].numerical.reshape(-1).float()
        # Probability-space BCE is evaluated in float32 outside autocast.
        truth=t.tensor(context.target.iloc[qi].to_numpy(np.float32),device='cuda')
        loss=t.nn.functional.binary_cross_entropy(probability.clamp(1e-6,1-1e-6),truth)
        penalty=sum((p-original[name]).square().sum()for name,p in selected.items())/384
        objective=loss+.001*penalty
        assert t.isfinite(objective)
        optimizer.zero_grad(set_to_none=True);objective.backward()
        assert all(p.grad is None for name,p in engine.model.named_parameters()if name not in selected)
        assert all(p.grad is None or t.isfinite(p.grad).all()for p in selected.values())
        optimizer.step()
        with t.no_grad():
            for name,p in selected.items():p.copy_(original[name]+(p-original[name]).clamp(-.25,.25))
        if (step+1)%25==0 or step==0:
            row=dict(step=step+1,training_loss=float(loss.detach()),objective=float(objective.detach()))
            history.append(row);emit('adaptation_training',fold=fold,**row)
            write(root/'training_progress.json',dict(fold=fold,episodes_done=step+1,total_episodes=200,history=history))
        if (step+1)%50==0:
            t.save(dict(selected={n:p.detach().cpu()for n,p in selected.items()},
                original={n:p.cpu()for n,p in original.items()},optimizer=optimizer.state_dict(),
                step=step+1,numpy_rng_state=rng.bit_generator.state),root/f'temperature/adapter_step_{step+1:03d}.pt')
    assert pretrained_hash(engine.model,selected)==original_hash
    max_change=max(float((p-original[n]).abs().max())for n,p in selected.items())
    engine.model.eval();gc.collect();t.cuda.empty_cache()
    emit('temperature_full_context_fit',fold=fold,completed_adaptation_episodes=200,max_temperature_update=max_change)
    adapted=engine.fit_predict(np.arange(50000),query,2026+fold,root/'temperature/fitted_context.pt')
    write(root/'temperature/cache.json',engine.last_cache)
    save_head(fold,'temperature',adapted,query,labels,indices,spline,head_columns)
    write(root/'boundary.json',dict(adaptation_label_rows=50000,correction_label_rows=6000,
        outer_validation_labels_available_to_worker=False,episodes=200,episode_support=384,episode_query=128,
        trainable_parameters=384,frozen_other_parameters_sha256=original_hash,
        maximum_parameter_update=max_change,peak_vram_bytes=t.cuda.max_memory_allocated(),
        maximum_raw_probability_change=float(np.max(abs(adapted-frozen))),seconds=time.time()-start))
    del engine,optimizer;gc.collect();t.cuda.empty_cache()
    # Independently read each completed cache once, retaining all checkpoints.
    for variant in ['frozen','temperature']:
        cache=read(root/variant/'cache.json');assert sha(root/variant/'fitted_context.pt')==cache['sha256']
    files={str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()and p.name!='complete.json'and p.name!='fitted_context.pt'}
    write(root/'complete.json',dict(status='complete',fold=fold,files=files,
        independently_verified_cache_sha256={v:read(root/v/'cache.json')['sha256']for v in ['frozen','temperature']}))
    emit('fold_complete',fold=fold,seconds=time.time()-start)
def main():
    plan=read(ROOT/'plan.json');assert plan['model_training_admitted'] is True
    assert read(ROOT/'full_shape_preflight.json')['status']=='passed'
    assert sha(plan['checkpoint'])==plan['checkpoint_sha256']
    (ROOT/'runs').mkdir(exist_ok=True);state=dict(original_deadline=plan['original_deadline'],completed_folds=[])
    for f in range(5):
        check(plan)
        if time.time()>=plan['original_deadline']-10800:
            state.update(status='partial',reason='Original finalization reserve; no reduced folds or steps');break
        state.update(status='running',active_fold=f);write(ROOT/'state.json',state)
        run_fold(f,plan);state['completed_folds'].append(f);write(ROOT/'state.json',state)
    else:state.update(status='training_complete',active_fold=None,scoring_host='pc2',outer_validation_labels_not_on_this_worker=True)
    write(ROOT/'state.json',state)
if __name__=='__main__':main()
