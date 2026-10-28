"""Finite five-fold CPU study, checkpointed without agent supervision."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
import argparse,fcntl,hashlib,json,time,sys,subprocess,signal,traceback,socket,platform
from pathlib import Path
import importlib.metadata
import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp,rankdata
import model

ROOT=Path(__file__).resolve().parent
DEADLINE=1792585871.823748
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def event(stage,**data):print(json.dumps(dict(time=time.time(),stage=stage,**data)),flush=True)
def deadline():assert time.time()<DEADLINE and not (ROOT/'STOP').exists(),'Original deadline or stop request'
def arrays(name):
    with np.load(ROOT/'input'/f'{name}.npz',allow_pickle=False) as z:return dict(z)
def metrics(y,p):
    assert np.isfinite(p).all() and np.all((p>0)&(p<1))
    order=np.argsort(p,kind='stable');end=np.r_[p[order][1:]!=p[order][:-1],True]
    cp=np.cumsum(y[order])[end];cn=np.cumsum(1-y[order])[end]
    ks=float(np.max(abs(cp/y.sum()-cn/(len(y)-y.sum()))))
    assert abs(ks-ks_2samp(p[y==1],p[y==0]).statistic)<1e-12
    n1=y.sum();auc=float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1)))
    return dict(KS=ks,AUC=auc,logloss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))))

def fold(f):
    deadline();dest=ROOT/'folds'/f'fold{f}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists(),'Preserve completed fit'
    rows=arrays('rows');ix=arrays(f'fold{f}');train,val=ix['train'],ix['validation'];n=len(rows['target'])
    assert len(train)==56000 and len(val)==14000 and not np.intersect1d(train,val).size
    # Feature fitting receives no default labels, IDs, or row-order predictors.
    data={key:arrays(key) for key in ['prices','bureau','income','repayment']};data['employment']=rows['employment']
    means,variance,missing,priors=model.measurement(data,train,n)
    x,v,transform=model.design(means,variance,missing,rows,train,read(ROOT/'input/manifest.json')['covariates'])
    write(dest/'measurement.json',priors);write(dest/'transform.json',transform)
    np.savez_compressed(dest/'posterior.npz',means=means,variance=variance,missing=missing)
    np.savez_compressed(dest/'design.npz',x=x,v=v,train=train,validation=val)
    event('fold_local_features_ready',fold=f,training_rows=len(train),query_rows=len(val),columns=x.shape[1])
    y=rows['target'][train]
    plugin,pm=model.fit(x[train],v[train],y,False);deadline()
    np.savez_compressed(dest/'plugin_model.npz',coefficients=plugin,fit_rows=train)
    write(dest/'plugin_optimizer.json',pm);event('plugin_fit_complete',fold=f,**pm)
    fitted,fm=model.fit(x[train],v[train],y,True,plugin);deadline()
    np.savez_compressed(dest/'integrated_model.npz',coefficients=fitted,fit_rows=train)
    write(dest/'integrated_optimizer.json',fm);event('integrated_fit_complete',fold=f,**fm)
    pred_plugin=expit(x[val]@plugin);pred_integrated=model.integrated(fitted,x[val],v[val])
    error=float(np.max(abs(pred_integrated-model.integrated(fitted,x[val],v[val],256))))
    assert error<=2e-5,('Quadrature resolution inadequate; retain as partial',error)
    ref=arrays('reference');base=ref['baseline'][val];kumo=ref['corrected_kumo'][val]
    predictions=dict(baseline=base,plugin_PD=pred_plugin,integrated_PD=pred_integrated,
        plugin_replacement=.75*kumo+.25*pred_plugin,integrated_replacement=.75*kumo+.25*pred_integrated,
        plugin_small_addition=.95*base+.05*pred_plugin,integrated_small_addition=.95*base+.05*pred_integrated)
    np.savez_compressed(dest/'predictions.npz',rows=val,**predictions)
    # Evaluation labels are first used only after both models and predictions exist.
    write(dest/'metrics.json',{name:metrics(rows['target'][val],p) for name,p in predictions.items()})
    write(dest/'boundary.json',dict(fit_rows=56000,validation_rows=14000,default_labels_used_for_measurement=False,
        training_ids_sha=hashlib.sha256('\n'.join(rows['application_id'][train]).encode()).hexdigest(),
        query_ids_sha=hashlib.sha256('\n'.join(rows['application_id'][val]).encode()).hexdigest(),
        quadrature_max_error_128_vs_256=error))
    write(dest/'complete.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'}))
    event('fold_complete',fold=f,quadrature_error=error)

def bootstrap(y,p,q,fold_ids,repetitions=1000):
    structure=[]
    for scores in [p,q]:
        order=np.argsort(scores,kind='stable');end=np.r_[scores[order][1:]!=scores[order][:-1],True]
        structure.append((order,end))
    groups=[np.flatnonzero((fold_ids==f)&(y==c)) for f in range(5) for c in [0,1]]
    rng=np.random.default_rng(20261029);draws=[]
    for rep in range(repetitions):
        if rep%100==0:deadline()
        w=np.zeros(len(y),int)
        for group in groups:w[group]=rng.multinomial(len(group),np.full(len(group),1/len(group)))
        n1=w[y==1].sum();n0=w[y==0].sum();m=[]
        for order,end in structure:
            cp=np.cumsum(w[order]*y[order])[end];cn=np.cumsum(w[order]*(1-y[order]))[end]
            dp=np.diff(np.r_[0,cp]);dn=np.diff(np.r_[0,cn])
            m.append(np.array([np.max(abs(cp/n1-cn/n0)),np.sum(dp*(cn-.5*dn))/(n1*n0)]))
        draws.append(m[0]-m[1])
    return np.asarray(draws)

def finish():
    rows=arrays('rows');y=rows['target'];n=len(y);seen=np.zeros(n,int);fold_ids=np.full(n,-1)
    preds={};fold_metrics={};replay_error=0.
    for f in range(5):
        d=ROOT/'folds'/f'fold{f}'
        for name,h in read(d/'complete.json')['files'].items():assert sha(d/name)==h
        z=dict(np.load(d/'predictions.npz'));val=z.pop('rows');seen[val]+=1;fold_ids[val]=f
        design=dict(np.load(d/'design.npz'))
        assert np.array_equal(design['validation'],val)
        assert not np.intersect1d(design['train'],val).size
        for variant in ['plugin','integrated']:
            fit=dict(np.load(d/(variant+'_model.npz')))
            assert np.array_equal(fit['fit_rows'],design['train'])
            repeated=(expit(design['x'][val]@fit['coefficients']) if variant=='plugin' else
                      model.integrated(fit['coefficients'],design['x'][val],design['v'][val]))
            error=float(np.max(abs(repeated-z[variant+'_PD'])));assert error<1e-10
            replay_error=max(replay_error,error)
        for name,p in z.items():
            if name not in preds:preds[name]=np.full(n,np.nan);fold_metrics[name]=[]
            preds[name][val]=p;fold_metrics[name].append(metrics(y[val],p))
    assert np.all(seen==1)
    scores={name:metrics(y,p) for name,p in preds.items()}
    assert abs(scores['baseline']['KS']-.5292994659005608)<1e-12
    comparisons={};draws={}
    pairs=[('primary','integrated_replacement','baseline'),('uncertainty_effect','integrated_replacement','plugin_replacement'),
           ('secondary_small_addition','integrated_small_addition','baseline')]
    for label,candidate,control in pairs:
        dd=bootstrap(y,preds[candidate],preds[control],fold_ids);draws[label]=dd
        delta={k:scores[candidate][k]-scores[control][k] for k in ['KS','AUC','logloss']}
        wins=sum(a['KS']>b['KS'] for a,b in zip(fold_metrics[candidate],fold_metrics[control]))
        comparisons[label]=dict(candidate=candidate,control=control,delta=delta,KS_fold_wins=wins,
            adjusted_conditional_interval=np.quantile(dd,[.004166666667,.995833333333],axis=0).tolist())
    primary=scores['integrated_replacement'];base=scores['baseline']
    result=dict(status='complete',metrics=scores,fold_metrics=fold_metrics,comparisons=comparisons,
        predeclared_primary='integrated_replacement',point_target_with_AUC_safeguard=bool(primary['KS']>=.54 and primary['AUC']>=base['AUC']),
        fitted_models=10,OOF_rows=n,complete_folds=5,max_prediction_replay_error=replay_error,
        interval_columns=['KS','AUC'],interval_scope='1000 paired fold/class bootstrap, Bonferroni six reported metric contrasts; conditional on predictions, not historical-selection or model-retraining uncertainty',
        scientific_limitations=['Repeated historical use of all evaluation labels','Independent Gaussian measurement assumptions are approximations',
            'Repayment snapshot timing unresolved','No causal inference or guarantee of public/private gain'],submission_generated=False)
    np.savez_compressed(ROOT/'oof.npz',target=y,fold=fold_ids,**preds)
    np.savez_compressed(ROOT/'bootstrap.npz',**draws);write(ROOT/'result.json',result)
    lines=['# Measurement uncertainty study','','Ten fitted actuarial GLMs; full five-fold 70k evaluation.','',
           '| Procedure | KS | AUC | Log loss |','|---|---:|---:|---:|']
    for name,m in scores.items():lines.append(f'| {name} | {m["KS"]:.8f} | {m["AUC"]:.8f} | {m["logloss"]:.8f} |')
    lines+=['','Primary comparison: integrated replacement versus retained incumbent. No blend-weight search.',
        f'.54 point target with AUC safeguard reached: {result["point_target_with_AUC_safeguard"]}.',
        'Models, raw sufficient statistics, posterior variances, priors and transformations are retained.',
        'The familiar labels are reused; conditional bootstrap intervals do not establish fresh generalization.',
        'No submission CSV was created.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');event('report_complete',primary=primary)

def control():
    import shutil
    lock=(ROOT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seal=read(ROOT/'seal.json')
    for name,h in seal['files'].items():assert sha(ROOT/name)==h,name
    for name,h in read(ROOT/'input/manifest.json')['files'].items():assert sha(ROOT/'input'/name)==h,name
    state=read(ROOT/'state.json') if (ROOT/'state.json').exists() else dict(deadline=DEADLINE,started=time.time())
    assert state['deadline']==DEADLINE and state.get('status')!='complete'
    for pid in state.get('active',{}).values():
        path=Path(f'/proc/{pid}/cmdline')
        assert not path.exists() or str(ROOT).encode() not in path.read_bytes(),'Existing owned worker'
    state.update(status='preflight',active={});write(ROOT/'state.json',state)
    packages={name:importlib.metadata.version(name) for name in ['numpy','scipy']}
    assert packages==read(ROOT/'plan.json')['versions']
    assert read(ROOT/'LEAKAGE_BOUNDARY.json')['status']=='passed'
    write(ROOT/'environment.json',dict(host=socket.gethostname(),python=platform.python_version(),packages=packages))
    write(ROOT/'preflight.json',model.preflight())
    pending=[]
    for f in range(5):
        path=ROOT/'folds'/f'fold{f}'/'complete.json'
        if path.exists():
            for name,h in read(path)['files'].items():assert sha(path.parent/name)==h
        else:pending.append(f)
    active={};state['status']='training'
    try:
        while pending or active:
            deadline();assert shutil.disk_usage(ROOT).free>2*1024**3
            while pending and len(active)<5:
                f=pending.pop(0);log=(ROOT/f'fold{f}.log').open('ab')
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'run.py'),'fold','--fold',str(f)],stdout=log,stderr=log,start_new_session=True)
                active[f]=(child,log);event('fold_started',fold=f,pid=child.pid)
            state['active']={str(f):p.pid for f,(p,_) in active.items()};write(ROOT/'state.json',state)
            for f,(p,log) in list(active.items()):
                if p.poll() is not None:
                    log.close();del active[f];assert p.returncode==0,f'Fold {f} failed; see fold{f}.log'
            time.sleep(3)
        state.update(status='reporting',active={});write(ROOT/'state.json',state);finish()
        state.update(status='complete',finished=time.time());write(ROOT/'state.json',state)
        event('DONE',status='complete',point_target_with_AUC_safeguard=read(ROOT/'result.json')['point_target_with_AUC_safeguard'])
    except BaseException:
        for p,log in active.values():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
        state.update(status='failed',active={},error=traceback.format_exc());write(ROOT/'state.json',state)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','fold','preflight','finish']);parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.action=='run':control()
    elif args.action=='fold':fold(args.fold)
    elif args.action=='preflight':print(json.dumps(model.preflight()))
    else:finish()
