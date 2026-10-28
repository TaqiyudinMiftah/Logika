"""One retained-state follow-up; does not refit teachers or Kumo contexts."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import argparse,sys,json,time,subprocess,gc
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.special import logit
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'plan.json').read_text());PARENT=Path(PLAN['parent'])
sys.path.insert(0,str(PARENT/'source'))
import baseline as b
from gpu_backend import Engine
from metaexp.features import Bank
from metaexp.pipeline import predict
EXTRA=['learned_horizontal_risk_logit','learned_horizontal_disagreement']

def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):b.dump(obj,path)
def deadline():assert time.time()<read(ROOT/'state.json')['deadline'] and not (ROOT/'STOP').exists()
def emit(stage,**values):print(json.dumps(dict(time=time.time(),stage=stage,**values)),flush=True)
def data():
    inp=Path(PLAN['input']);frame=pd.read_parquet(inp/'data/train_features.parquet')
    bank=Bank(Path(PLAN['bank']).parent)
    assert frame.application_id.tolist()==bank.metadata.application_id.iloc[:70000].tolist()
    return inp,frame,bank,read(inp/'BASELINE_LOCK.json'),read(inp/'provenance/definition.json')
def base_frame(frame,context,cf):
    result=frame.copy()
    for j,col in enumerate(EXTRA):result[col]=np.nan;result.loc[context,col]=cf[:,j]
    return result
def metrics(y,p):
    p=np.asarray(p,float);y=np.asarray(y,int);assert np.isfinite(p).all()
    m=b.scores(y,p);m['logloss']=float(log_loss(y,p))
    assert abs(m['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-12
    n1=y.sum();auc=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
    assert abs(m['AUC']-auc)<1e-12
    return m
def preflight():
    while True:
        deadline();state=read(PARENT/'state.json')
        if state['status']=='complete':break
        assert state['status']!='failed','Parent failed; retain the dependency failure'
        status=subprocess.check_output(['systemctl','--user','show',
            'logika-ri[REDACTED_CREDENTIAL].service','-p','ActiveState','--value'],text=True).strip()
        assert status in ['active','activating'],'Parent is not active; do not duplicate its jobs'
        time.sleep(15)
    subprocess.run([sys.executable,str(PARENT/'verify_completed.py')],check=True)
    audit=read(PARENT/'INDEPENDENT_AUDIT.json')
    assert audit['status']=='passed'
    if audit['point_target_with_AUC_safeguard']:
        dump(ROOT/'preflight.json',dict(status='passed',skip=True,reason='Parent target independently verified'))
        return
    inp,frame,bank,lock,spec=data();directory=PARENT/'folds/fold0'
    z=np.load(directory/'teacher_features.npz');raw=np.load(directory/'signal/prediction.npz')
    cache=directory/'signal/fitted_context.pt';assert b.sha(cache)==read(directory/'signal/complete.json')['files']['fitted_context.pt']
    context=base_frame(frame,z['context'],z['context_features'])
    query=frame.iloc[z['query'][:512]].drop(columns='target').copy()
    for j,col in enumerate(EXTRA):query[col]=z['query_features'][:512,j]
    engine=Engine(PLAN['checkpoint'],lock['Kumo']['features']+EXTRA,context)
    replay=engine.replay(cache,query);error=float(np.max(abs(replay-raw['raw'][:512])))
    assert error<2e-5,error
    import torch
    gpu=dict(name=torch.cuda.get_device_name(),peak_vram=torch.cuda.max_memory_allocated())
    del engine;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'preflight.json',dict(status='passed',skip=False,parent_independently_verified=True,
         original_query_cache_replay_max_error=error,accelerator=gpu,teacher_training_repeated=False))
    emit('preflight_pass',replay_error=error)
def fold(f):
    deadline();inp,frame,bank,lock,spec=data();parent=PARENT/f'folds/fold{f}'
    dest=ROOT/f'folds/fold{f}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        for name,digest in read(dest/'complete.json')['files'].items():assert b.sha(dest/name)==digest
        return
    z=np.load(parent/'teacher_features.npz');context=z['context'];query=z['query']
    assert len(context)==50000 and len(query)==20000 and not np.intersect1d(context,query).size
    assignment=np.empty(len(query),int)
    assignment[np.random.default_rng(20261027+f).permutation(len(query))]=np.arange(len(query))%5
    components=np.full((len(query),5),np.nan);seen=np.zeros(len(query),int)
    for j in range(5):
        take=np.flatnonzero(assignment==j);rows=query[take]
        for c in range(5):
            deadline();folder=parent/f'teachers/inner{j}/teacher{c}'
            for name,digest in read(folder/'complete.json')['files'].items():assert b.sha(folder/name)==digest
            bundle=joblib.load(folder/'model.joblib');pre=bundle['preprocessing'];fit=pre.fit_rows
            assert len(fit)==40000 and not np.intersect1d(fit,query).size
            assert set(fit).issubset(set(context));pre.bank=bank
            components[take,c]=predict(bundle['model'],pre.transform(rows),bundle['weight'])
        seen[take]+=1
        emit('query_features_complete',fold=f,inner_teacher_group=j,rows=len(rows))
    assert np.all(seen==1) and np.isfinite(components).all()
    trimmed=np.sort(components,axis=1)[:,1:4].mean(axis=1)
    features=np.column_stack([logit(np.clip(trimmed,1e-6,1-1e-6)),
        logit(np.clip(components,1e-6,1-1e-6)).std(axis=1)]).astype(np.float32)
    np.savez_compressed(dest/'query_features.npz',rows=query,assignment=assignment,
                        component_predictions=components,features=features)
    cache=parent/'signal/fitted_context.pt';expected=read(parent/'signal/complete.json')['files']['fitted_context.pt']
    assert b.sha(cache)==expected
    context_frame=base_frame(frame,context,z['context_features'])
    query_frame=frame.iloc[query].drop(columns='target').copy()
    for j,col in enumerate(EXTRA):query_frame[col]=features[:,j]
    engine=Engine(PLAN['checkpoint'],lock['Kumo']['features']+EXTRA,context_frame)
    emit('Kumo_cached_replay',fold=f,rows=len(query))
    raw=engine.replay(cache,query_frame)
    head=b.refit_correction(spec,frame.iloc[query[:6000]],raw[:6000])
    sp=pd.read_parquet(inp/f'artifacts/fold{f}/spline_oof.parquet').prediction.to_numpy()
    final=.75*b.corrected_probability(head,frame.iloc[query[6000:]].drop(columns='target'),raw[6000:])[0]+.25*sp
    joblib.dump(head,dest/'correction.joblib',compress=3)
    np.savez_compressed(dest/'predictions.npz',rows=query,raw=raw,final=final)
    dump(dest/'provenance.json',dict(teacher_fit_rows=40000,query_assignment_seed=20261027+f,
        query_labels_used_for_features=False,correction_rows=6000,
        reused_context_path=str(cache),reused_context_sha=expected,
        teacher_fits=0,Kumo_fits=0,context_ids_sha=b.ids_hash(frame,context),query_ids_sha=b.ids_hash(frame,query)))
    dump(dest/'complete.json',dict(files={p.name:b.sha(p) for p in dest.iterdir() if p.is_file()}))
    emit('outer_complete',fold=f)
def finish():
    deadline();inp,frame,bank,lock,spec=data();y=frame.target.to_numpy(int)
    old=np.load(PARENT/'oof.npz');pred=np.full(len(y),np.nan);seen=np.zeros(len(y),int)
    for f in range(5):
        folder=ROOT/f'folds/fold{f}'
        for name,digest in read(folder/'complete.json')['files'].items():assert b.sha(folder/name)==digest
        z=np.load(folder/'predictions.npz');rows=z['rows'][6000:]
        original=np.load(inp/f'artifacts/fold{f}/indices.npz');assert np.array_equal(rows,original['validation'])
        head=joblib.load(folder/'correction.joblib')
        sp=pd.read_parquet(inp/f'artifacts/fold{f}/spline_oof.parquet').prediction.to_numpy()
        replay=.75*b.corrected_probability(head,frame.iloc[rows].drop(columns='target'),z['raw'][6000:])[0]+.25*sp
        assert np.array_equal(replay,z['final']);pred[rows]=replay;seen[rows]+=1
    assert np.all(seen==1)
    all_predictions={'matched_query':pred,**{name:old[name] for name in ['signal','null','historical','baseline']}}
    m={name:metrics(y,p) for name,p in all_predictions.items()}
    folds={name:[metrics(y[old['fold']==f],p[old['fold']==f]) for f in range(5)] for name,p in all_predictions.items()}
    structures={}
    for name,p in all_predictions.items():
        order=np.argsort(p,kind='stable');structures[name]=order,np.r_[p[order][1:]!=p[order][:-1],True]
    groups=[np.flatnonzero((old['fold']==f)&(y==c)) for f in range(5) for c in [0,1]]
    rng=np.random.default_rng(20261028);draws={name:[] for name in ['signal','null','historical']}
    for rep in range(PLAN['bootstrap_repetitions']):
        if rep%200==0:deadline();emit('bootstrap',completed=rep)
        w=np.zeros(len(y),int)
        for group in groups:w[group]=rng.multinomial(len(group),np.full(len(group),1/len(group)))
        n1=w[y==1].sum();n0=w[y==0].sum();values={}
        for name,(order,end) in structures.items():
            cp=np.cumsum(w[order]*(y[order]==1))[end];cn=np.cumsum(w[order]*(y[order]==0))[end]
            dp=np.diff(np.r_[0,cp]);dn=np.diff(np.r_[0,cn])
            values[name]=np.array([np.max(abs(cp/n1-cn/n0)),np.sum(dp*(cn-.5*dn))/(n1*n0)])
        for name in draws:draws[name].append(values['matched_query']-values[name])
    interval={name:np.quantile(values,[.004166666667,.995833333333],axis=0).tolist() for name,values in draws.items()}
    point=bool(m['matched_query']['KS']>=.54 and m['matched_query']['AUC']>=m['historical']['AUC'])
    result=dict(status='complete',metrics=m,fold_metrics=folds,point_goal_reached=point,
        paired_KS_AUC_intervals=interval,teacher_fits=0,Kumo_fits=0,retained_contexts_replayed=5,
        new_correction_fits=5,OOF_rows=70000,historical_label_reuse=True,
        uncertainty_scope='2000 paired fold/class bootstrap, Bonferroni six metrics/comparisons; conditional on fitted predictions, not corrected for historical adaptive selection',
        submission_generated=False,point_in_time_certified=False)
    dump(ROOT/'result.json',result)
    np.savez_compressed(ROOT/'oof.npz',target=y,fold=old['fold'],**all_predictions)
    lines=['# Matched teacher training size','',f'.54 KS target with AUC safeguard: {point}.','',
           '| Pipeline | KS | AUC | Log loss |','|---|---:|---:|---:|']
    for name,score in m.items():lines.append(f'| {name} | {score["KS"]:.8f} | {score["AUC"]:.8f} | {score["logloss"]:.8f} |')
    lines+=['','All 70k development rows have one outer prediction. All models and cache references retained.',
        'Only query teacher training size changed; no teacher or backbone was retrained.',
        'All labels have historical exposure. No private leaderboard guarantee or competition CSV is claimed.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');emit('DONE',point_goal_reached=point)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','fold','finish']);p.add_argument('--root');p.add_argument('--fold',type=int);a=p.parse_args()
    if a.action=='preflight':preflight()
    elif a.action=='fold':fold(a.fold)
    else:finish()
