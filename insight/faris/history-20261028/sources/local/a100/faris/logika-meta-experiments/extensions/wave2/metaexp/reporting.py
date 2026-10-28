from __future__ import annotations
import csv
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from .common import read,write,sha,identity,metrics,verify_oof,FAMILIES

def audit(root,workspace):
    root=Path(root).resolve();workspace=Path(workspace).resolve();records=[];sources={};seen={}
    art=workspace/'Logika/artifacts'
    for path in sorted(art.rglob('*.json')):
        obj=read(path);rel=str(path.relative_to(workspace));sources[rel]=sha(path)
        if not isinstance(obj,list):continue
        for row in obj:
            if not isinstance(row,dict) or 'ks' not in row:continue
            signature=identity(row);name=str(row.get('name',row.get('recipe',row.get('version','unnamed'))))
            if signature in seen:
                records[seen[signature]]['references'].append(rel);continue
            seen[signature]=len(records)
            records.append({'id':'history_'+signature[:16],'name':name,'state':'reported_but_unverified','result':row,'references':[rel],
                            'reason':'Stored summary; training logs, split identities and OOF must verify before using as measured baseline.'})
    history_files=list((workspace/'a100/faris').glob('*history*.json'))+list((root/'history_sources').glob('*history*.json'))
    for path in history_files:
        obj=read(path);sources[str(path.relative_to(workspace))]=sha(path)
        if isinstance(obj,list):
            for rec in obj:
                if rec.get('state')=='verified_runtime_only':rec['state']='verified';rec['scope']='runtime_only'
                records.append(rec)
    # Attach exact definitions; a definition without result evidence is planned,
    # rather than silently being treated as completed or never attempted.
    by_name={r['name']:r for r in records}
    for path in sorted((art/'optimization').rglob('definitions*.json')):
        definitions=read(path);definitions=definitions if isinstance(definitions,list) else list(definitions.values())
        for spec in definitions:
            if not isinstance(spec,dict) or 'name' not in spec:continue
            names=[spec['name']+'_'+str(stage) for stage in spec['stages']] if 'stages' in spec else [spec['name']]
            for name in names:
                if name in by_name:by_name[name]['configuration']=spec
                else:records.append({'id':'planned_'+identity({'file':str(path.relative_to(art)),'name':name})[:16],'name':name,'state':'planned','configuration':spec,'references':[str(path.relative_to(workspace))],'reason':'Definition exists; matching completed result evidence was not found by this importer.'})
    log=workspace/'a100/projects/logika-ui-26/EXPERIMENT_LOG.md'
    if log.exists():sources[str(log.relative_to(workspace))]=sha(log)
    public=read(art/'optimization/public_scores.json')
    out={'records':records,'sources':sources,'public_references':public,
         'state_definitions':{'verified':'Evidence independently checked; scope must be read alongside state.','reported_but_unverified':'Result reported without a complete independent replay.','partial':'Evidence of an unfinished attempt.','failed':'Attempt terminated with an error.','planned':'Configuration registered without completed evidence.','no_evidence_found':'No matching record found; absence is not proof of no attempt.'},
         'coverage_assessments':[{'name':name,'state':'no_evidence_found','scope':'systematic matched grid in inspected historical records'} for name in ['encoding/imputation combinations','broad reduction comparisons','cross-model matched feature contrasts','nested learned ensemble comparisons']],
         'historical_final_metadata':{v:read(art/'optimization'/v/f'submission_{v}_metadata.json') for v in ['v4','v5']},
         'node380_kumo':{'state':'verified','ks':0.5127016592084279,'oof_sha256':'b73bb8f52b9a4857d05350edbb26501d55b573bfcf05fbe6a9115132647b6b3c',
                         'verification':'Independently recomputed from 70,000 saved OOF rows during planning; held submission only.'},
         'known_families':['source ablation','repayment shrinkage','bureau smoothing','cash-flow coverage','category prices','splines','specialists','shallow trees','TabM','latent variables','fixed ensembles'],
         'coverage_gaps':'No systematic matching evidence found for the new full encoding, imputation, reduction and nested-ensemble grids. This does not prove no prior attempt exists.'}
    write(root/'history_inventory.json',out);return out

def verify_run(root,tid,bank):
    folder=Path(root)/'runs'/tid;complete=read(folder/'complete.json')
    for rel,h in complete['files'].items():
        from .common import inside
        if sha(inside(folder,rel))!=h:raise ValueError('Artifact checksum differs: '+tid+'/'+rel)
    score=read(folder/'scores.json')
    if score.get('bank_id')!=bank.manifest['bank_id']:raise ValueError('Different feature bank')
    if (Path(root)/'campaign.json').exists():
        expected=read(Path(root)/'campaign.json')['code_hash']
        if expected!='fixture' and score.get('code_hash')!=expected:raise ValueError('Different run source')
    kind=score['kind']
    if kind in ['pipeline','reference','fixed_ensemble','nested_ensemble']:
        p=np.load(folder/'prediction.npy');actual=verify_oof(bank.y,p,bank.folds)
        if abs(actual['pooled_oof']['ks']-score['pooled_oof']['ks'])>1e-12:raise ValueError('Saved OOF metric differs')
        if abs(fast_ks(bank.y,p)-score['pooled_oof']['ks'])>1e-12:raise ValueError('Independent KS implementation differs')
        if (folder/'oof.npz').exists():
            o=np.load(folder/'oof.npz')
            if not np.array_equal(o['target'],bank.y) or not np.array_equal(o['fold'],bank.folds):raise ValueError('OOF labels or folds differ')
            if not np.array_equal(o['application_id'],bank.metadata.application_id.iloc[bank.dev].to_numpy(dtype=str)):raise ValueError('OOF ID order differs')
        if kind=='pipeline':
            task=read(folder/'task.json')
            for f in range(3):
                a=read(folder/f'fold_{f}/preprocessing.json');ti=bank.dev[bank.folds!=f]
                if a['fit_rows_hash']!=identity(ti.tolist()):raise ValueError('Transform fitted outside outer training')
    elif kind=='confirmation':
        p=np.load(folder/'prediction.npy');y=np.load(bank.root/'confirmation_y.npy')
        if abs(metrics(y,p)['ks']-score['metrics']['ks'])>1e-12:raise ValueError('Confirmation metric differs')
    return score

def eda(root,bank):
    import pandas as pd
    from scipy.stats import ks_2samp
    root=Path(root);rng=np.random.default_rng(2026);fit=bank.dev
    test=np.flatnonzero(bank.metadata.split.eq('test').to_numpy())
    sample=rng.choice(fit,min(len(fit),12000),replace=False);rows=[]
    for i,name in enumerate(bank.names):
        v=np.asarray(bank.numeric[sample,i]);t=np.asarray(bank.numeric[test,i]);a=v[np.isfinite(v)];b=t[np.isfinite(t)]
        rows.append({'feature':name,'missing_development':float(np.mean(~np.isfinite(v))),'missing_test':float(np.mean(~np.isfinite(t))),
                     'mean':float(a.mean()) if len(a) else None,'std':float(a.std()) if len(a) else None,
                     'p01':float(np.quantile(a,.01)) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p99':float(np.quantile(a,.99)) if len(a) else None,
                     'train_test_ks':float(ks_2samp(a,b,method='asymp').statistic) if len(a) and len(b) else None,
                     'source':','.join(bank.manifest['info'][name]['sources'])})
    dest=root/'reports';dest.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(dest/'feature_eda.csv',index=False)
    cohorts=[]
    for col in bank.categorical:
        values=bank.categorical[col].iloc[fit].to_numpy()
        for group in np.unique(values):
            ix=values==group;cohorts.append({'column':col,'group':str(group),'development_n':int(ix.sum()),'development_rate':float(bank.y[ix].mean())})
    pd.DataFrame(cohorts).to_csv(dest/'development_cohorts.csv',index=False)
    pd.to_datetime(bank.metadata.decision_datetime.iloc[fit]).dt.to_period('M').astype(str).value_counts().sort_index().rename_axis('decision_month').rename('development_rows').to_csv(dest/'temporal_coverage.csv')
    availability=[]
    for source in ['application','profile','previous','repayment','bureau','account','shopping']:
        ix=[i for i,n in enumerate(bank.names) if bank.manifest['info'][n]['sources']==[source]]
        if not ix:continue
        present=np.zeros(len(sample),dtype=bool)
        for j in ix:present|=np.isfinite(bank.numeric[sample,j])
        availability.append({'source':source,'sample_rows':len(sample),'at_least_one_recorded_feature_fraction':float(present.mean()),'caveat':'Finite feature availability; zero-filled historical counts can encode absent histories.'})
    write(dest/'source_availability.json',availability)
    ix=np.argsort([-float(np.nanstd(bank.numeric[sample,j])) if np.isfinite(bank.numeric[sample,j]).any() else 0 for j in range(len(bank.names))])[:128]
    matrix=np.array(bank.numeric[sample[:,None],ix[None,:]],dtype=float)
    for j in range(matrix.shape[1]):
        v=matrix[:,j];v[~np.isfinite(v)]=np.nanmedian(v) if np.isfinite(v).any() else 0
    with np.errstate(invalid='ignore',divide='ignore'):corr=np.nan_to_num(np.corrcoef(matrix,rowvar=False))
    pairs=[{'feature_a':bank.names[ix[a]],'feature_b':bank.names[ix[b]],'correlation':float(corr[a,b])} for a in range(len(ix)) for b in range(a+1,len(ix)) if abs(corr[a,b])>=.9]
    pd.DataFrame(pairs,columns=['feature_a','feature_b','correlation']).to_csv(dest/'high_correlations.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].hist([r['missing_development'] for r in rows],bins=30);ax[0].set_title('Feature missingness, development only')
    ax[1].hist([r['train_test_ks'] for r in rows if r['train_test_ks'] is not None],bins=30);ax[1].set_title('Unlabeled train/test feature drift')
    fig.tight_layout();fig.savefig(dest/'eda.png',dpi=130);plt.close(fig)
    write(dest/'eda_summary.json',{'rows':bank.manifest['shape'][0],'numeric_features':len(bank.names),'categorical_features':len(bank.categorical.columns),
                                 'development_rows':len(bank.dev),'confirmation_rows_reserved':bank.manifest['confirmation_rows'],
                                 'development_positive_rate':float(bank.y.mean()),'confirmation_labels_opened':False,
                                 'raw_audits':'Bank construction enforces point-in-time, completed months, unique IDs and order grain. See builder source and raw hashes.'})

def fast_ks(y,p):
    order=np.argsort(-p,kind='stable');labels=np.asarray(y)[order];values=np.asarray(p)[order]
    last=np.r_[np.flatnonzero(np.diff(values)!=0),len(values)-1]
    tp=np.cumsum(labels)[last];fp=last+1-tp
    return float(np.max(np.abs(tp/np.sum(labels)-fp/(len(labels)-np.sum(labels)))))

def final_statistics(root,until):
    root=Path(root);finals=read(root/'finalists.json');y=np.load(root/'bank/confirmation_y.npy');baseline=root/'runs/c_reference_v3/prediction.npy'
    if not baseline.exists():
        result={'state':'unavailable','reason':'Matched v3 confirmation baseline did not finish','primary_nominee':finals.get('primary')};write(root/'reports/confirmation.json',result);return result
    b=np.load(baseline);records=[];rng=np.random.default_rng(20261016);zero=np.flatnonzero(y==0);one=np.flatnonzero(y==1)
    ids=[t for t in finals['ids'] if t!='reference_v3' and (root/'runs'/('c_'+t)/'complete.json').exists()]
    arrays={t:np.load(root/'runs'/('c_'+t)/'prediction.npy') for t in ids};samples={t:[] for t in ids}
    for i in range(500):
        if time.time()>until:break
        ix=np.r_[rng.choice(zero,len(zero),replace=True),rng.choice(one,len(one),replace=True)];bk=fast_ks(y[ix],b[ix])
        for t,p in arrays.items():samples[t].append(fast_ks(y[ix],p[ix])-bk)
    for tid,p in arrays.items():
        values=np.array(samples[tid]);delta=fast_ks(y,p)-fast_ks(y,b)
        rec={'id':tid,'ks':fast_ks(y,p),'baseline_ks':fast_ks(y,b),'delta':delta,'bootstrap_samples':len(values)}
        if len(values)>=100:
            rec.update(ci95=np.quantile(values,[.025,.975]).tolist(),one_sided_p=float((1+sum(values<=0))/(len(values)+1)))
        else:rec.update(ci95=None,one_sided_p=None)
        records.append(rec)
    candidates=sorted([r for r in records if r['one_sided_p'] is not None],key=lambda r:r['one_sided_p']);reject=True
    for i,r in enumerate(candidates):
        reject=reject and r['one_sided_p']<=.05/(len(candidates)-i);r['holm_significant_positive']=bool(reject and r['delta']>0)
    result={'state':'completed','primary_nominee':finals['primary'],'baseline':'reference_v3','historical_holdout_caveat':'Reserved from this campaign only; historical experiments used these labels.',
            'records':records,'uncertainty':'500 stratified paired bootstrap resamples, bounded by verification deadline; Holm correction across completed frozen candidates.'}
    write(root/'reports/confirmation.json',result);return result

def report(root,final=False):
    import pandas as pd
    root=Path(root);from .features import Bank
    bank=Bank(root);cfg=read(root/'campaign.json');catalogue=read(root/'catalogue.json');dest=root/'reports';dest.mkdir(exist_ok=True)
    accepted={};rejected=[];states={}
    for p in sorted((root/'runs').glob('*/status.json')):
        tid=p.parent.name;s=read(p);states[tid]=s
        if s['state']!='completed':continue
        try:accepted[tid]=verify_run(root,tid,bank)
        except Exception as e:rejected.append({'id':tid,'error':str(e)})
    scoreboards={k:[] for k in ['pipeline','reference','fixed_ensemble','nested_ensemble','repeat','temporal','confirmation']}
    for tid,s in accepted.items():
        row={'id':tid,**s};scoreboards.setdefault(s['kind'],[]).append(row)
    for kind,rows in scoreboards.items():
        rows.sort(key=lambda r:r.get('pooled_oof',r.get('metrics',{})).get('ks',-1),reverse=True);write(dest/('scoreboard_'+kind+'.json'),rows)
    controls={}
    for tid,s in accepted.items():
        if s['kind']=='pipeline' and s['representation']==catalogue['control']:controls[(s['model'],s['device'])]=(tid,s)
    effects=[]
    for tid,s in accepted.items():
        if s['kind']!='pipeline':continue
        control=controls.get((s['model'],s['device']))
        if not control:continue
        cid,b=control;g=[a['ks']-v['ks'] for a,v in zip(s['per_fold'],b['per_fold'])];delta=s['pooled_oof']['ks']-b['pooled_oof']['ks']
        effect='positive_cv' if delta>=.001 and sum(v>0 for v in g)>=2 else 'negative_cv' if delta<=-.001 and sum(v<0 for v in g)>=2 else 'mixed_or_small'
        task=read(root/'runs'/tid/'task.json');changes=task['recipe']['representation']['changes']
        effects.append({'id':tid,'family':s['family'],'model':s['model'],'device':s['device'],'control':cid,'ks_delta':delta,'fold_gains':json.dumps(g),'effect':effect,
                        'changes':json.dumps(changes,sort_keys=True),'single_factor_contrast':len(changes)==1})
    pd.DataFrame(effects).to_csv(dest/'feature_effects.csv',index=False)
    errors=[]
    for (_,device),(tid,s) in controls.items():
        prediction=np.load(root/'runs'/tid/'prediction.npy')
        for column in bank.categorical:
            values=bank.categorical[column].iloc[bank.dev].to_numpy()
            for group in np.unique(values):
                mask=values==group
                if len(np.unique(bank.y[mask]))<2:continue
                errors.append({'control':tid,'device':device,'column':column,'group':str(group),'rows':int(mask.sum()),'target_rate':float(bank.y[mask].mean()),'mean_prediction':float(prediction[mask].mean()),**metrics(bank.y[mask],prediction[mask])})
    pd.DataFrame(errors).to_csv(dest/'baseline_errors.csv',index=False)
    registered={r['id']:r for r in catalogue['recipes']};tested={s['recipe_id'] for s in accepted.values() if s['kind']=='pipeline'}
    coverage={f:{'registered':sum(r['representation']['family']==f for r in registered.values()),'completed':sum(registered[t]['representation']['family']==f for t in tested)} for f in FAMILIES}
    pending=[{'id':r['id'],'family':r['representation']['family'],'model':r['model'],'representation':r['representation']['id']} for r in catalogue['recipes'] if r['id'] not in tested]
    write(dest/'untested_recipes.json',pending);write(dest/'coverage.json',{'families':coverage,'design_coverage':catalogue['coverage']})
    counts={k:sum(v['state']==k for v in states.values()) for k in sorted({v['state'] for v in states.values()})}
    assigned=set();assigned_unknown=[]
    if (root/'ledger.sqlite').exists():
        with sqlite3.connect('file:'+str(root/'ledger.sqlite')+'?mode=ro',uri=True) as db:
            for tid,rid,state in db.execute("SELECT id,recipe_id,state FROM tasks WHERE mode='cv'"):
                assigned.add(rid)
                if state in ['assigned','running'] and tid not in states:assigned_unknown.append(tid)
    summary={'campaign':cfg['id'],'started':cfg['started'],'deadline':cfg['deadline'],'generated':time.time(),'final':final,
             'completed_distinct_pipelines':len(tested),'completed_fixed_ensembles':len(scoreboards['fixed_ensemble']),'completed_nested_ensembles':len(scoreboards['nested_ensemble']),
             'registered_pipelines':len(registered),'uncompleted_pipelines':len(pending),'attempt_states':counts,'verification_rejections':rejected,
             'unassigned_pipelines':len(set(registered)-assigned) if (root/'ledger.sqlite').exists() else None,'assigned_without_collected_terminal_evidence':assigned_unknown,
             'source_hash':cfg['code_hash'],'bank_id':bank.manifest['bank_id'],'confirmation_opened':bool(scoreboards['confirmation'])}
    write(dest/'summary.json',summary)
    best=scoreboards['pipeline'][:20];handoff=[]
    for s in best:
        task=read(root/'runs'/s['id']/'task.json');handoff.append({'recipe':task['recipe'],'cv':s['pooled_oof'],'device':s['device'],
            'preprocessing_audits':[str((root/'runs'/s['id']/f'fold_{f}/preprocessing.json').relative_to(root)) for f in range(3)],
            'resources':read(root/'runs'/s['id']/'resources.json') if (root/'runs'/s['id']/'resources.json').exists() else {'elapsed_seconds':s.get('elapsed_seconds')},
            'feature_lineage':'bank/manifest.json:info','factory':'metaexp.pipeline.Preprocessor',
            'reproduce':'Use the frozen recipe in catalogue.json with metaexp.pipeline.fit_predict(bank, recipe, fit_rows, y_fit, query_rows, device, seed); fit every transform on fit_rows.'})
    write(dest/'a100_handoff.json',{'recipes':handoff,'source_hash':cfg['code_hash'],'bank_id':bank.manifest['bank_id'],
        'scope':'Reusable feature factories and measured probe evidence. Refit cohort transforms on A100 training partitions. No guaranteed foundation-model gain.'})
    lines=[f'# Campaign {cfg["id"]}',f'Final report: {final}. Elapsed hours: {(time.time()-cfg["started"])/3600:.3f}.',
           f'Completed distinct fitted pipelines: {len(tested)} / {len(registered)}. Fixed ensembles: {len(scoreboards["fixed_ensemble"])}. Nested ensembles: {len(scoreboards["nested_ensemble"])}.',
           f'Attempt states: {counts}. Rejected evidence: {len(rejected)}.',
           'Protocol: 56,000 development applications, full 3-fold OOF; 14,000 reserved for this campaign. Historical work used these labels.',
           'Historical public-best Logika v3 score 0.55845 is not directly comparable with this campaign CV.',
           '## Best fitted probes','| Recipe | Family | Model | OOF KS | Mean fold KS |','|---|---|---|---:|---:|']
    for s in best[:10]:lines.append(f'| {s["id"]} | {s["family"]} | {s["model"]} | {s["pooled_oof"]["ks"]:.6f} | {s["mean_fold_ks"]:.6f} |')
    lines += ['## Evidence and limits','See feature_effects.csv for matched per-model/backend changes, coverage.json for factor coverage and untested_recipes.json for unfinished work.',
              'Positive/negative CV labels are exploratory, including multiple-change combinations. Confirmation and repeated seeds provide separate evidence.',
              'No prediction-test labels, leaderboard queries, foundation-model runs or competition submissions are part of this campaign.']
    for title,group in [('Largest positive matched CV effects',sorted([e for e in effects if e['ks_delta']>0],key=lambda e:e['ks_delta'],reverse=True)[:5]),('Largest negative matched CV effects',sorted([e for e in effects if e['ks_delta']<0],key=lambda e:e['ks_delta'])[:5])]:
        lines.append('## '+title)
        lines.extend(f'- {e["id"]}: {e["model"]}/{e["device"]}, delta KS {e["ks_delta"]:+.6f}; {e["changes"]}.' for e in group)
    if (dest/'confirmation.json').exists():
        confirmation=read(dest/'confirmation.json');lines.append('## Frozen confirmation')
        lines.append('Primary nominee selected before confirmation: '+str(confirmation.get('primary_nominee')))
        for row in confirmation.get('records',[]):lines.append(f'- {row["id"]}: delta KS {row["delta"]:+.6f}, 95% paired bootstrap CI {row.get("ci95")}, Holm significant positive: {row.get("holm_significant_positive",False)}.')
    (dest/'final_report.md' if final else dest/'progress.md').write_text('\n\n'.join(lines)+'\n')
    lessons=[f'# Measured effects, {cfg["id"]}']
    lessons += [f'- {e["id"]}: {e["effect"]}, delta KS={e["ks_delta"]:.6f}, {e["model"]}/{e["device"]}, {e["changes"]}' for e in sorted(effects,key=lambda e:e['ks_delta'],reverse=True)[:30]]
    (root/'learnings.md').write_text('\n'.join(lessons)+'\n')
    return summary
