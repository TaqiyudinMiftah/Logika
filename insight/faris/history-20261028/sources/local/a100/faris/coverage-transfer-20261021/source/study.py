"""One coverage mechanism, matched ablations and fixed opportunity extensions."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
import feature_head
from build import MONTHS,AUX,NEW

ROOT=Path(__file__).resolve().parents[1]
INPUT=Path('/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/input')
def read(p):return json.loads(Path(p).read_text())
def emit(stage,**values):print(json.dumps(dict(time=time.time(),stage=stage,**values)),flush=True)
def check():assert time.time()<read(ROOT/'plan.json')['original_deadline'] and not (ROOT/'STOP').exists()
def metrics(y,p):
    p=np.asarray(p,float);assert np.isfinite(p).all() and np.all((p>0)&(p<1))
    result=b.scores(y,p);result['logloss']=float(log_loss(y,p))
    assert abs(result['KS']-ks_2samp(p[y==1],p[y==0]).statistic)<1e-12
    n1=y.sum();auc=float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1)))
    assert abs(result['AUC']-auc)<1e-12
    return result
def load():
    data=pd.read_parquet(INPUT/'data/train_features.parquet');features=pd.read_parquet(ROOT/'data/features.parquet')
    assert data.application_id.tolist()==features.application_id.tolist()
    for col in MONTHS+AUX+NEW:assert col not in data;data[col]=features[col].to_numpy()
    assert len(data)==70000 and data.application_id.is_unique
    base=read(INPUT/'BASELINE_LOCK.json')['correction']['features'];v25=base+MONTHS+AUX
    recipes=dict(original=base,support_only=base+AUX,months_only=base+MONTHS,V25=v25)
    for col in MONTHS+AUX:recipes['drop_'+col]=[c for c in v25 if c!=col]
    recipes.update(opportunity_primary=v25+NEW,no_allocation_only=v25+NEW[:2],mixed_allocation_only=v25+NEW[2:])
    assert len(recipes)==13 and len({tuple(v) for v in recipes.values()})==13
    assert list(recipes)==read(ROOT/'plan.json')['recipes']
    return data,recipes
def inputs(data,f):
    directory=INPUT/f'artifacts/fold{f}';ix=dict(np.load(directory/'indices.npz'))
    assert len(ix['correction'])==6000 and len(ix['context'])==50000 and len(ix['validation'])==14000
    assert not np.intersect1d(ix['context'],ix['correction']).size
    assert not np.intersect1d(ix['train'],ix['validation']).size
    meta=read(directory/'metadata.json')
    for name,label in [('correction','correction_training'),('context','context'),('validation','validation')]:
        assert b.ids_hash(data,ix[name])==meta[label+'_ids_sha256']
    predictions={}
    for name,split in [('kumo_holdout','correction'),('kumo_oof','validation'),('spline_oof','validation'),('final_oof','validation')]:
        frame=pd.read_parquet(directory/(name+'.parquet'))
        assert frame.application_id.tolist()==data.application_id.iloc[ix[split]].tolist()
        predictions[name]=frame.prediction.to_numpy(float)
    return ix,predictions
def preflight():
    check();assert read(ROOT/'data/AUDIT.json')['status']=='passed'
    manifest=read(ROOT/'evidence/import_manifest.json')
    for name,digest in manifest['files'].items():assert b.sha(ROOT/'evidence'/name)==digest,name
    data,recipes=load();all_v25=np.full(len(data),np.nan);seen=np.zeros(len(data),int);checks=[]
    for f in range(5):
        ix,p=inputs(data,f);v=data.iloc[ix['validation']].drop(columns='target')
        model=joblib.load(ROOT/f'evidence/runs/V25/fold{f}_correction.joblib')
        assert model['columns']==recipes['V25']
        actual=.75*b.corrected_probability(model,v,p['kumo_oof'])[0]+.25*p['spline_oof']
        saved=pd.read_parquet(ROOT/f'evidence/runs/V25/fold{f}_validation.parquet')
        assert saved.application_id.tolist()==v.application_id.tolist()
        error=float(np.max(abs(actual-saved.prediction)));assert error<1e-12,(f,error)
        changed=v.iloc[:64].copy();changed['target']=999;changed['customer_id']='POISONED'
        poisoned=.75*b.corrected_probability(model,changed,p['kumo_oof'][:64])[0]+.25*p['spline_oof'][:64]
        assert np.array_equal(poisoned,actual[:64])
        all_v25[ix['validation']]=actual;seen[ix['validation']]+=1
        checks.append(dict(fold=f,replay_max_error=error,outer_label_poisoning='passed'))
    assert np.all(seen==1)
    y=data.target.to_numpy(int);m=metrics(y,all_v25);ref=read(ROOT/'evidence/runs/V25/RESULT.json')['OOF']
    for metric in ['KS','AUC','logloss']:assert abs(m[metric]-ref[metric])<1e-12
    b.dump(dict(status='passed',OOF_rows=70000,checks=checks,verified_V25_metrics=m,
        raw_features_independently_rebuilt=True,selected_default_labels_used_for_feature_construction=False,
        public_score=.56822,public_source='User report for exact SHA256-identified CSV; not independently queried from Kaggle',
        historical_labels_reused=True),ROOT/'preflight.json')
    emit('preflight_pass',V25=m)
def fold(f):
    check();data,recipes=load();ix,p=inputs(data,f)
    dest=ROOT/f'folds/fold{f}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists(),'Completed fold already exists'
    training=data.iloc[ix['correction']].copy();valid=data.iloc[ix['validation']].drop(columns='target').copy()
    predictions={};diagnostics={}
    for name,cols in recipes.items():
        check();model=feature_head.fit(training,p['kumo_holdout'],cols,.003,.5)
        joblib.dump(model,dest/(name+'.joblib'),compress=3)
        head,_=b.corrected_probability(model,valid,p['kumo_oof']);pred=.75*head+.25*p['spline_oof']
        repeated=.75*b.corrected_probability(joblib.load(dest/(name+'.joblib')),valid,p['kumo_oof'])[0]+.25*p['spline_oof']
        assert np.array_equal(pred,repeated)
        if name=='original':assert np.max(abs(pred-p['final_oof']))<2e-6
        if name=='V25':
            saved=pd.read_parquet(ROOT/f'evidence/runs/V25/fold{f}_validation.parquet')
            assert np.max(abs(pred-saved.prediction))<2e-6
        predictions[name]=pred
        diagnostics[name]=dict(columns=cols,parameters=len(model['coefficients']),gradient_max=model['gradient_max_abs'],
            iterations=model['iterations'],training_rows=6000)
        emit('head_complete',fold=f,recipe=name)
    # Exact retained controls distinguish estimator refit rounding from changes.
    predictions['published_V25']=pd.read_parquet(ROOT/f'evidence/runs/V25/fold{f}_validation.parquet').prediction.to_numpy()
    predictions['published_original']=p['final_oof']
    np.savez_compressed(dest/'predictions.npz',rows=ix['validation'],**predictions)
    b.dump(diagnostics,dest/'diagnostics.json')
    b.dump(dict(status='complete',files={p.name:b.sha(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'},
        fit_ids_sha=b.ids_hash(data,ix['correction']),evaluation_ids_sha=b.ids_hash(data,ix['validation'])),dest/'complete.json')
    emit('fold_complete',fold=f,models=13)
def finish():
    check();data,recipes=load();y=data.target.to_numpy(int);seen=np.zeros(len(y),int);foldids=np.full(len(y),-1);preds={}
    for f in range(5):
        d=ROOT/f'folds/fold{f}'
        for name,digest in read(d/'complete.json')['files'].items():assert b.sha(d/name)==digest
        z=dict(np.load(d/'predictions.npz'));rows=z.pop('rows');seen[rows]+=1;foldids[rows]=f
        assert np.array_equal(rows,inputs(data,f)[0]['validation'])
        for name,p in z.items():
            if name not in preds:preds[name]=np.full(len(y),np.nan)
            preds[name][rows]=p
    assert np.all(seen==1)
    scores={name:metrics(y,p) for name,p in preds.items()}
    folds={name:[metrics(y[foldids==f],p[foldids==f]) for f in range(5)] for name,p in preds.items()}
    contrasts={'support_effect':('support_only','original'),'months_effect':('months_only','original'),
        'months_beyond_support':('V25','support_only'),'support_beyond_months':('V25','months_only'),
        'primary_opportunity_effect':('opportunity_primary','V25'),
        'no_allocation_effect':('no_allocation_only','V25'),'mixed_allocation_effect':('mixed_allocation_only','V25')}
    for col in MONTHS+AUX:contrasts['contribution_'+col]=('V25','drop_'+col)
    effects={}
    for name,(a,c) in contrasts.items():
        delta={k:scores[a][k]-scores[c][k] for k in ['KS','AUC','logloss']}
        fd=[u['KS']-v['KS'] for u,v in zip(folds[a],folds[c])]
        effects[name]=dict(candidate=a,control=c,delta=delta,KS_fold_wins=int(np.sum(np.array(fd)>0)),fold_KS_deltas=fd)
    # One predeclared primary bootstrap, including ties and class/fold strata.
    names=['opportunity_primary','V25'];structures={}
    for name in names:
        p=preds[name];order=np.argsort(p,kind='stable');end=np.r_[p[order][1:]!=p[order][:-1],True];structures[name]=order,end
    rng=np.random.default_rng(20261031);groups=[np.flatnonzero((foldids==f)&(y==c)) for f in range(5) for c in [0,1]];draws=[]
    for rep in range(2000):
        if rep%200==0:check();emit('primary_bootstrap',completed=rep)
        w=np.zeros(len(y),int)
        for group in groups:w[group]=rng.multinomial(len(group),np.full(len(group),1/len(group)))
        n1=w[y==1].sum();n0=w[y==0].sum();values={}
        for name,(order,end) in structures.items():
            cp=np.cumsum(w[order]*y[order])[end];cn=np.cumsum(w[order]*(1-y[order]))[end]
            dp=np.diff(np.r_[0,cp]);dn=np.diff(np.r_[0,cn])
            values[name]=np.array([np.max(abs(cp/n1-cn/n0)),np.sum(dp*(cn-.5*dn))/(n1*n0)])
        draws.append(values[names[0]]-values[names[1]])
    effects['primary_opportunity_effect']['conditional_95_interval']=np.quantile(draws,[.025,.975],axis=0).tolist()
    np.savez_compressed(ROOT/'oof.npz',target=y,fold=foldids,**preds)
    np.savez_compressed(ROOT/'primary_bootstrap.npz',delta=np.asarray(draws))
    result=dict(status='complete',metrics=scores,fold_metrics=folds,effects=effects,trained_heads=65,OOF_rows=70000,
        predeclared_primary='opportunity_primary',primary_KS_target_reached=bool(scores['opportunity_primary']['KS']>=.54),
        public_reference='V25 .56822 user-reported for verified CSV hash',
        AUC_policy='Reported as complementary ranking evidence, not an absolute veto based on the V25 public counterexample.',
        interval_columns=['KS','AUC'],secondary_contrasts='Descriptive post-selection attribution, not confirmatory tests',
        limitations=['Historical evaluation labels reused; public model chosen after leaderboard observation',
            'One public result does not establish universal OOF transfer','Inherited baseline snapshot provenance remains unresolved'],
        submission_generated=False)
    b.dump(result,ROOT/'result.json')
    pd.DataFrame([dict(recipe=name,**m) for name,m in scores.items()]).sort_values('KS',ascending=False).to_csv(ROOT/'scores.csv',index=False)
    lines=['# V25 coverage attribution and allocation opportunity','','Public reference: V25 .56822 (user report, exact CSV hash verified).','',
        '| Recipe | OOF KS | AUC |','|---|---:|---:|']
    for name,m in sorted(scores.items(),key=lambda kv:kv[1]['KS'],reverse=True):lines.append(f'| {name} | {m["KS"]:.8f} | {m["AUC"]:.8f} |')
    lines+=['',f'Primary .54 KS target reached: {result["primary_KS_target_reached"]}.',
        'All 65 correction heads and preprocessing are retained. Every recipe completed all five original folds.',
        'Only the fixed opportunity extension is primary. Attribution and secondary recipes are descriptive.',
        'These labels were previously used; no fresh private holdout or leaderboard improvement is claimed.',
        'No competition submission was created.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');emit('DONE',status='complete',primary=scores['opportunity_primary'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','fold','finish']);p.add_argument('--fold',type=int);a=p.parse_args()
    if a.action=='preflight':preflight()
    elif a.action=='fold':fold(a.fold)
    else:finish()
