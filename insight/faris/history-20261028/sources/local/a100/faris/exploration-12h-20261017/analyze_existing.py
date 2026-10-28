"""Post-campaign descriptive audit. Reads saved development OOF; never fits models.

Confirmation predictions/labels and competition-test predictions are not opened.
New combinations and cohort comparisons are post-hoc diagnostics, not promotions.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, collections, hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, Path): return str(x)
    return x


def main(root, output):
    root=Path(root).resolve(); output=Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    sys.path[:0]=[str(root/'source'), str(root)]
    from supervisor import records, original_champion_record
    from features import ExploreBank
    from utils import combine
    read=lambda p: json.loads(Path(p).read_text())
    def save(name, obj): (output/name).write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n')
    cfg=read(root/'source/configurations.json'); views=read(root/'source/views.json')
    dep=read(root/'deployment.json'); bankpath=Path(dep['hosts']['pc2']['bank'])
    bank=ExploreBank(bankpath.parent, root/'extras'); y=bank.y; folds=bank.folds; dev=bank.dev
    rows=records(); known={r['id'] for r in rows}
    for e in read(root/'evidence/node380-final-outcomes.json'):
        if e['id'] not in known:
            rows.append({'id':e['id'], 'host':'node380', 'job':e['job'],
                         'evaluation':e['outcome']['evaluation'], 'recovered_node':True})
    screen=[r for r in rows if r['job']['protocol']=='screen']
    lookup={r['id']:r for r in screen}; cache={}
    def pred(r):
        if r['id'] not in cache:
            path=(root/'evidence/node380-final-predictions'/(r['id']+'.npz')) if r.get('recovered_node') else Path(r['folder'])/'validation.npz'
            z=np.load(path); assert np.array_equal(z['rows'], dev); assert np.array_equal(z['fold'], folds)
            p=z['prediction']; assert p.shape==y.shape and np.isfinite(p).all(); cache[r['id']]=p
        return cache[r['id']]
    def ks(t, p):
        return float(ks_2samp(p[t==0], p[t==1]).statistic) if len(np.unique(t))==2 else None
    def score(p, t=y, f=folds):
        return {'ks':ks(t,p), 'fold_ks':[ks(t[f==j],p[f==j]) for j in range(3)],
                'auc':roc_auc_score(t,p), 'log_loss':log_loss(t,np.clip(p,1e-8,1-1e-8)),
                'brier':brier_score_loss(t,p), 'mean_probability':float(np.mean(p))}
    base=np.load(root/'evidence/champion_oof.npz')['existing_feature_ensemble']
    new=np.load(root/'reports/ens_e62cfb5b6d636145.npy')
    parents=read(root/'reports/frozen_finalists.json')['nominees'][0]['spec']['parents']
    old=sorted([r for r in screen if original_champion_record(r)], key=lambda r:r['job']['params']['config'])
    assert len(old)==5
    assert np.max(np.abs(np.sort(np.column_stack([pred(r) for r in old]),axis=1)[:,1:-1].mean(1)-base))<1e-12
    summary={'scope':'Existing development OOF only; no fitting, no confirmation access, post-hoc descriptive comparisons',
             'development_rows':len(y),'defaults':int(y.sum()), 'baseline':score(base),'primary':score(new),
             'completed_run_records':len(rows),'screening_records':len(screen)}

    # Family quality, instability, and declared search coverage. These are unequal searches.
    family=[]
    for f in sorted({r['job']['family'] for r in screen}):
        rr=[r for r in screen if r['job']['family']==f]; vals=np.array([r['evaluation']['score'] for r in rr]); best=max(rr,key=lambda r:r['evaluation']['score'])
        family.append({'family':f,'completed_records':len(rr),'unique_configurations':len({r['job']['params']['config'] for r in rr}),
                       'best_id':best['id'],'best_ks':vals.max(),'median_ks':np.median(vals),'p10_ks':np.quantile(vals,.1),
                       'degenerate_fold_runs':sum(any(s<.1 for s in r['evaluation'].get('per_fold',[])) for r in rr),
                       'best_configuration':cfg.get(best['job']['params']['config']),
                       'search_is_unbalanced':True})
    save('families.json', sorted(family,key=lambda x:x['best_ks'],reverse=True))

    # Prediction diversity for family leaders, old components and new components.
    chosen={r['id']:r for r in old}
    for r in screen:
        f=r['job']['family']
        if r['id'] in parents or r['id']==next(x['best_id'] for x in family if x['family']==f): chosen[r['id']]=r
    arrays={k:pred(r) for k,r in chosen.items()}; arrays['champion']=base; arrays['new_logit']=new
    ids=list(arrays); x=np.column_stack([arrays[k] for k in ids]); ranks=np.column_stack([rankdata(x[:,j]) for j in range(x.shape[1])]); corr=np.corrcoef(ranks,rowvar=False)
    pd.DataFrame(corr,index=ids,columns=ids).to_csv(output/'prediction_spearman.csv')
    top={k:np.argsort(-p,kind='stable')[:len(y)//5] for k,p in arrays.items()}; comparison=[]
    for k,p in arrays.items():
        if k=='champion': continue
        mask=np.zeros(len(y),bool); mask[top[k]]=True; bm=np.zeros(len(y),bool);bm[top['champion']]=True
        comparison.append({'id':k,'spearman_with_champion':corr[ids.index(k),ids.index('champion')],
                           'top20_jaccard':len(set(top[k])&set(top['champion']))/len(set(top[k])|set(top['champion'])),
                           'defaults_in_top20':int(y[mask].sum()),'champion_defaults_in_top20':int(y[bm].sum()),
                           'added_defaults':int(y[mask&~bm].sum()),'lost_defaults':int(y[bm&~mask].sum()), **score(p)})
    save('diversity.json',comparison)
    for label, keys in [('old',[r['id'] for r in old]),('new',parents)]:
        ii=[ids.index(k) for k in keys]; c=corr[np.ix_(ii,ii)]; off=c[np.triu_indices(len(ii),1)]
        summary[label+'_rank_redundancy']={'mean_pair_spearman':off.mean(),'min_pair_spearman':off.min(),
                  'correlation_participation_ratio':np.trace(c)**2/np.sum(c*c),
                  'participation_ratio_is_not_an_estimate_of_predictive_information':True}
    summary['nested_pair_spearman']=corr[ids.index('pc2_job_000853'),ids.index('pc2_job_000855')]

    # Frozen-catalogue ensemble results; no fresh selection on confirmation.
    fixed=read(root/'reports/scoreboard_fixed.json'); positive=[r for r in fixed if r['delta']>0]
    summary['fixed_catalogue']={'evaluated':len(fixed),'positive_vs_champion':len(positive),
                              'positive_and_fold_gate':sum(r['eligible'] for r in positive),
                              'unique_component_sets_among_positive':len({tuple(r['spec']['parents']) for r in positive}),
                              'by_method':{m:{'count':sum(r['spec']['method']==m for r in fixed),'positive':sum(r['spec']['method']==m for r in positive)} for m in sorted({r['spec']['method'] for r in fixed})}}

    # Small, explicitly post-hoc fixed-combination diagnostics on three EXISTING CV evaluations.
    combination_tests=[]
    for protocol in ['screen','reseed','repeat']:
        ff=folds if protocol!='repeat' else np.asarray(read(root/'evidence/repeat.json')['folds'])
        a={}
        for k in parents:
            rr=lookup[k]
            matches=[r for r in rows if r['job']['protocol']==protocol and r['host']==rr['host'] and r['job']['family']==rr['job']['family'] and r['job']['params']==rr['job']['params']]
            if not matches:break
            z=np.load(Path(matches[-1]['folder'])/'validation.npz'); assert np.array_equal(z['rows'],dev);assert np.array_equal(z['fold'],ff);a[k]=z['prediction']
        if len(a)!=len(parents):continue
        b=base if protocol=='screen' else np.load(root/'reports'/('champion_'+protocol+'.npy'))
        full=combine({'method':'logit','parents':parents},a)
        tests={'full_new':full}
        for k in parents:tests['without_'+k]=combine({'method':'logit','parents':[p for p in parents if p!=k]},a)
        for w in [.1,.25,.5]:
            tests['old_new_probability_'+str(w)]=(1-w)*b+w*full
            z=np.clip(np.column_stack([b,full]),1e-8,1-1e-8);tests['old_new_logit_'+str(w)]=1/(1+np.exp(-np.log(z/(1-z))@np.array([1-w,w])))
        for name,p in tests.items():
            combination_tests.append({'protocol':protocol,'diagnostic':name,'ks':ks(y,p),'delta_vs_champion':ks(y,p)-ks(y,b),
                                      'delta_vs_full_new':ks(y,p)-ks(y,full),
                                      'fold_delta_vs_champion':[ks(y[ff==j],p[ff==j])-ks(y[ff==j],b[ff==j]) for j in range(3)],
                                      'posthoc_not_confirmed':True})
    save('fusion_and_ablation.json',combination_tests)

    # Inspect feature treatments: same model does not mean a single changed feature block.
    contrasts=read(root/'reports/feature_effects.json'); contrasts_out=[]
    for c in contrasts:
        a=lookup[c['treatment']];b=lookup[c['control']];ac=cfg[a['job']['params']['config']];bc=cfg[b['job']['params']['config']]
        av=views[ac['view']]['config'];bv=views[bc['view']]['config'];diff={k:[bv.get(k),av.get(k)] for k in set(av)|set(bv) if av.get(k)!=bv.get(k)}
        contrasts_out.append({**c,'same_host':a['host']==b['host'],'changed_config_fields':diff,'changed_field_count':len(diff)})
    save('feature_contrasts_audited.json',contrasts_out)
    summary['feature_contrast_design']={'reported_contrasts':len(contrasts),'same_host':sum(c['same_host'] for c in contrasts_out),
             'multi_field_changes':sum(c['changed_field_count']>1 for c in contrasts_out),
             'single_field_changes':sum(c['changed_field_count']==1 for c in contrasts_out),
             'zero_field_changes':sum(c['changed_field_count']==0 for c in contrasts_out)}

    # Effective execution signatures use actual input columns and stored fitted model parameters.
    # Candidate duplicate groups are confirmed by exact saved probability equality, not names.
    signatures=collections.defaultdict(list); raw_feature_counter=collections.Counter(); feature_use=[]; optimizer=[]
    def fingerprint(r):
        j=r['job']; p=cfg.get(j['params']['config']);f=j['family']
        if not p or f=='champion':return None
        rep=views[p['view']];c=rep['config']; names,cats=bank.columns(rep)
        learned={k:c[k] for k in ['missing','clip','indicator','transform','select','feature_count','reducer','spline']}
        learned['scale']='standard' if c['scale']=='none' and (p['model'] in ['act_logit','act_cloglog','woe_scorecard','cred_scorecard','elastic','spline','rff','tabm','resmlp','ft','tabr','gru','deepsets'] or c['reducer']!='none') else c['scale']
        if c['select']=='correlation':learned['correlation']=c['correlation']
        if c['reducer']!='none':learned.update(dimensions=c['dimensions'],append=c['append'])
        if cats:learned['encoding']=c['encoding']
        if any(bank.manifest['info'].get(n,{}).get('price') for n in names):learned['category_normalize']=c['category_normalize']
        count_pairs=any(n.endswith('_k') and n[:-2]+'_n' in names for n in names)
        if count_pairs:learned.update(strength=c['strength'],representation=c['representation'])
        path=Path(r['folder'])/'fold_0/parameters.json' if r.get('folder') else None
        if not path or not path.exists():return None
        actual=read(path);sig={'host':r['host'],'family':f,'seed':j['seed'],'columns':names,'cats':cats,'preprocessing':learned,'model':actual,'weight':p['weight']}
        return hashlib.sha256(json.dumps(sig,sort_keys=True,default=str).encode()).hexdigest()
    for r in screen:
        if r.get('folder') and r['job']['family']!='nested':
            fp=fingerprint(r)
            if fp:signatures[fp].append(r)
            file=Path(r['folder'])/'fold_0/preprocessing.json'
            if file.exists():
                a=read(file);names=a.get('selected_features',[]);raw_feature_counter.update(names)
                feature_use.append({'id':r['id'],'family':r['job']['family'],'ks':r['evaluation']['score'],
                                    'selected_count':len(names),'new_selected':sum(n.startswith('new_') for n in names),
                                    'output_features':a.get('output_features'),'actual_selection':a.get('actual_selection'),'actual_reducer':a.get('actual_reducer')})
            if r['job']['family'] in ['act_logit','act_cloglog']:
                for fold in range(3):
                    f=Path(r['folder'])/f'fold_{fold}/actuarial.json'
                    if f.exists():
                        a=read(f);optimizer.append({'id':r['id'],'family':r['job']['family'],'fold':fold,'success':a.get('optimizer_success'),'iterations':a.get('iterations')})
    exact_groups=[]; total_candidates=0
    for group in signatures.values():
        if len(group)<2:continue
        total_candidates+=len(group)-1; checks=collections.defaultdict(list)
        for r in group:
            z=np.load(Path(r['folder'])/'validation.npz');p=z['prediction'];checks[hashlib.sha256(p.tobytes()).hexdigest()].append(r['id'])
        for identical in checks.values():
            if len(identical)>1:exact_groups.append(identical)
    save('execution_duplicates.json',{'conservative_signature_candidates':total_candidates,'bit_identical_groups':exact_groups,
         'redundant_completed_records':sum(len(g)-1 for g in exact_groups),'no_retraining':True})
    save('selected_feature_usage.json',{'records':feature_use,'selection_frequency_is_not_causal_importance':True,'top_selected':raw_feature_counter.most_common(60)})
    save('optimizer_status.json',optimizer)

    # Redundancy of the 214 new feature columns versus the original feature bank.
    sample=np.random.default_rng(20261018).choice(dev,min(4000,len(dev)),replace=False)
    aa=np.array(bank.extra[sample],dtype=float);bb=np.array(bank.numeric[sample],dtype=float)
    def standardized_ranks(z):
        constant=[]
        for j in range(z.shape[1]):
            a=z[:,j];valid=np.isfinite(a);fill=np.median(a[valid]) if valid.any() else 0;a[~valid]=fill
            if np.ptp(a)<1e-12:constant.append(j)
            z[:,j]=rankdata(a)
        z-=z.mean(0);den=np.sqrt(np.sum(z*z,axis=0));z/=np.where(den>0,den,1);return z,constant
    aa,constant=standardized_ranks(aa);bb,_=standardized_ranks(bb);cor=aa.T@bb;nearest=np.argmax(np.abs(cor),axis=1);redundancy=[]
    for j,name in enumerate(bank.extra_names):
        n=bank.names[nearest[j]];col=np.abs(cor[j,nearest[j]]);identical=False
        if col>.999999:
            left=np.asarray(bank.extra[dev,j]);right=np.asarray(bank.numeric[dev,nearest[j]]);identical=bool(np.allclose(left,right,rtol=1e-5,atol=1e-6,equal_nan=True))
        redundancy.append({'new':name,'family':bank.extra_meta['info'][name]['family'],'nearest_original':n,
                           'absolute_rank_correlation':col,'constant_in_sample':j in constant,'verified_equal_on_full_development':identical})
    save('feature_redundancy.json',redundancy)

    # Fixed, target-independent cohorts diagnose where different experts could matter.
    def col(name):return np.asarray(bank.numeric[dev,bank.lookup[name]],dtype=float)
    cohorts={}
    for label, mask in [('account_missing',col('has_account')==0),('account_present',col('has_account')>0),
                        ('repayment_missing',col('has_repayments')==0),('repayment_present',col('has_repayments')>0),
                        ('bureau_0',col('bureau_provider_count')==0),('bureau_1',col('bureau_provider_count')==1),('bureau_2plus',col('bureau_provider_count')>=2),
                        ('repayment_1to5',(col('repayment_installment_count')>=1)&(col('repayment_installment_count')<=5)),
                        ('repayment_6to20',(col('repayment_installment_count')>5)&(col('repayment_installment_count')<=20)),
                        ('repayment_21plus',col('repayment_installment_count')>20)]:cohorts[label]=mask
    ratio=col('observed_declared_income_ratio')
    for label, mask in [('observed_income_low',ratio<.5),('observed_income_similar',(ratio>=.5)&(ratio<=1.5)),('observed_income_high',ratio>1.5),('income_ratio_missing',~np.isfinite(ratio))]:cohorts[label]=mask
    months=bank.metadata.decision_datetime.iloc[dev].astype(str).str[:7].to_numpy()
    for month in np.unique(months):cohorts['decision_'+month]=months==month
    cohort_results=[]
    cohort_experts=['champion','new_logit']+parents+[x['best_id'] for x in family if x['family'] in ['kumo','tabicl','act_logit','act_cloglog','xgb','cat_native','gru','deepsets']]
    for name,mask in cohorts.items():
        n=int(mask.sum());pos=int(y[mask].sum())
        if min(pos,n-pos)<30:continue
        for k in dict.fromkeys(cohort_experts):
            p=arrays[k];d=[]
            for f in range(3):
                ix=mask&(folds==f)
                if min(int(y[ix].sum()),int((1-y[ix]).sum()))<10:d.append(None)
                else:d.append(ks(y[ix],p[ix])-ks(y[ix],base[ix]))
            cohort_results.append({'cohort':name,'n':n,'defaults':pos,'expert':k,'ks':ks(y[mask],p[mask]),
                                   'delta_vs_champion':ks(y[mask],p[mask])-ks(y[mask],base[mask]),
                                   'fold_delta':d,'brier':brier_score_loss(y[mask],p[mask]),
                                   'log_loss':log_loss(y[mask],np.clip(p[mask],1e-8,1-1e-8)),
                                   'posthoc_not_a_validated_gate':True})
    save('cohorts.json',cohort_results)

    # Diagnose numerical failures using saved scorecard transforms, without refitting.
    import joblib
    scorecard_diagnostics=[]
    for fam in ['woe_scorecard','cred_scorecard']:
        rr=sorted([r for r in screen if r['job']['family']==fam],key=lambda r:r['evaluation']['score'])
        for r in [rr[0],rr[-1]]:
            folder=Path(r['folder'])/'fold_0';m=joblib.load(folder/'model.joblib');pre=joblib.load(folder/'preprocessing.joblib');pre.bank=bank;pre.pre.bank=bank
            ix=dev[folds==0][:2000];xx=pre.transform(ix);mapped=m._apply(xx[:,m.columns_],m.maps_);z=m.scale_.transform(mapped);prob=m.predict_proba(xx)[:,1]
            parts={}
            for j,name in enumerate(['woe','uncertainty','credibility'] if fam=='cred_scorecard' else ['woe']):
                cols=np.arange(j,z.shape[1],3 if fam=='cred_scorecard' else 1);parts[name]={'median_absolute_z':float(np.median(np.abs(z[:,cols]))),
                          'fraction_abs_z_above_8':float(np.mean(np.abs(z[:,cols])>8)),
                          'l1_coefficient_mass':float(np.abs(m.model_.coef_[0,cols]).sum())}
            scorecard_diagnostics.append({'id':r['id'],'family':fam,'oof_ks':r['evaluation']['score'],'sample_rows':len(ix),
                'prediction_unique':len(np.unique(prob)),'fraction_prediction_below_1e_6':float(np.mean(prob<1e-6)),
                'fraction_prediction_above_1_minus_1e_6':float(np.mean(prob>1-1e-6)),
                'mapped_columns':parts,'scaler_min_scale':float(m.scale_.scale_.min()),
                'diagnosis':'Outer-validation covariates transformed by saved full-training maps versus scaler fitted on inner-OOF maps; no new fitting'})
    save('scorecard_diagnostics.json',scorecard_diagnostics)
    save('summary.json',summary)
    save('sources.json',{'retrieved':'2026-10-18','sources':[
       {'title':'Cawley and Talbot 2010, model-selection overfitting','url':'https://www.jmlr.org/beta/papers/v11/cawley10a.html'},
       {'title':'Jacobs et al. 1991, Adaptive Mixtures of Local Experts','url':'https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf'}]})
    print(json.dumps(clean({'summary':summary,'execution_duplicate_groups':len(exact_groups),
                            'redundant_completed_records':sum(len(g)-1 for g in exact_groups),
                            'new_features_rankcorr_ge_095':sum(r['absolute_rank_correlation']>=.95 for r in redundancy),
                            'new_features_equal_existing':sum(r['verified_equal_on_full_development'] for r in redundancy)}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.root,a.output)
