"""Model-only experiments on fixed V25 inputs and frozen prediction caches.

The batch name integrates with the existing experiment registry; it generates
no new historical features. Outer labels are used only for held-out evaluation.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import fcntl
import gc
import json
from pathlib import Path
import sys
import time
import traceback
import urllib.request
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, brier_score_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
V = ROOT/'v7'
sys.path[:0] = [str(ROOT/'scripts'),str(V/'scripts')]
from baseline import dump, sha, scores, corrected_probability, design, ids_hash
from experiments import combined, fit_head, bootstrap

TARGET = .531


def state(status, **kwargs):
    dump(dict(status=status,pid=os.getpid(),updated=time.time(),target=TARGET,**kwargs),V/'reports/MODELING_QUEUE.json')


def metric(y,p):
    return dict(scores(y,p),logloss=float(log_loss(y,p)),brier=float(brier_score_loss(y,p)))


def load():
    parent=json.loads((V/'runs/V25/RESULT.json').read_text())
    columns=parent['columns'];schema=pq.read_schema(ROOT/'data/train_features.parquet').names
    basecols=[c for c in columns if c in schema]
    meta=[c for c in ['application_id','target','decision_datetime'] if c in schema]
    tr=pd.read_parquet(ROOT/'data/train_features.parquet',columns=list(dict.fromkeys(meta+basecols)))
    te=pd.read_parquet(ROOT/'data/test_features.parquet',columns=[c for c in list(dict.fromkeys(meta+basecols)) if c!='target'])
    addon=[c for c in columns if c not in basecols]
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    y=tr.target.to_numpy(int);fold_ids=np.full(len(tr),-1);basepred=np.full(len(tr),np.nan);frames=[];cache_checks=[]
    for fold in range(5):
        a=ROOT/'artifacts'/f'fold{fold}';idx=np.load(a/'indices.npz');hi,vi,ci=idx['correction'],idx['validation'],idx['context']
        assert len(set(np.r_[hi,vi,ci]))==70000
        assert (len(hi),len(vi),len(ci))==(6000,14000,50000)
        frameset={}
        for part,base in [('correction',tr.iloc[hi]),('validation',tr.iloc[vi]),('test',te)]:
            ext=pd.read_parquet(V/'extension/data'/f'fold{fold}'/(part+'.parquet'),columns=['application_id']+addon)
            frameset[part]=combined(base,ext)
        h,val,test=[frameset[k] for k in ['correction','validation','test']]
        kh,kv,kt=[pd.read_parquet(a/f'kumo_{part}.parquet') for part in ['holdout','oof','test']]
        sv,st=[pd.read_parquet(a/f'spline_{part}.parquet') for part in ['oof','test']]
        for frame,cached in [(h,kh),(val,kv),(test,kt),(val,sv),(test,st)]:
            assert frame.application_id.tolist()==cached.application_id.tolist()
        model=joblib.load(V/'runs/V25'/f'fold{fold}_correction.joblib')
        assert model['columns']==columns
        # Retain only the6k training basis; validation/test bases are transient.
        basis={'correction':design(model,h,kh.prediction)}
        ph,_=corrected_probability(model,h,kh.prediction);pv,_=corrected_probability(model,val,kv.prediction)
        old=pd.read_parquet(V/'runs/V25'/f'fold{fold}_validation.parquet')
        assert np.max(abs(.75*pv+.25*sv.prediction.to_numpy()-old.prediction))<2e-12
        fold_ids[vi]=fold;basepred[vi]=old.prediction
        priors=dict(context=float(y[ci].mean()),correction=float(y[hi].mean()),outer_training=float(y[np.r_[ci,hi]].mean()),source='Only50k context and6k correction labels; no outer-validation or test labels')
        frames.append(dict(fold=fold,hi=hi,vi=vi,h=h,val=val,test=test,kh=kh.prediction.to_numpy(),kv=kv.prediction.to_numpy(),kt=kt.prediction.to_numpy(),sv=sv.prediction.to_numpy(),st=st.prediction.to_numpy(),head=model,basis=basis,training_priors=priors))
        cache_checks.append(dict(fold=fold,context_ids_sha256=ids_hash(tr,ci),correction_ids_sha256=ids_hash(tr,hi),validation_ids_sha256=ids_hash(tr,vi),V25_replay_max_abs_error=float(np.max(abs(.75*pv+.25*sv.prediction.to_numpy()-old.prediction)))))
    assert np.isfinite(basepred).all()
    dump(dict(status='PASS',folds=cache_checks,V25_submission_sha256=parent['submission_sha256'],features_unchanged=True,backbone_frozen=1.),V/'modeling/reports/CACHE_AUDIT.json')
    return tr,te,sample,y,fold_ids,basepred,frames,parent


def top_set(p,fraction):
    k=int(round(len(p)*fraction));mask=np.zeros(len(p),bool)
    mask[np.argsort(-np.asarray(p),kind='stable')[:k]]=True
    return mask


def error_analysis(out,tr,y,folds,reference,pred,frames):
    analysis=[];mask20=top_set(reference,.2)
    for fraction in [.1,.2,.3236857142857143,.4,.5]:
        a,b=top_set(reference,fraction),top_set(pred,fraction);entrants=b&~a;exits=a&~b
        analysis.append(dict(top_population_fraction=fraction,reference_bad_captured=int(y[a].sum()),candidate_bad_captured=int(y[b].sum()),net_bad_captured=int(y[b].sum()-y[a].sum()),new_members=int(entrants.sum()),exited_members=int(exits.sum()),entrant_bad_rate=float(y[entrants].mean()) if entrants.any() else None,exit_bad_rate=float(y[exits].mean()) if exits.any() else None))
    feature_names=['aux_account_months','aux_allocation_count','risk_conditional_negative_months','risk_conditional_nonnegative_months','installment_observed_income_ratio','re_late_30_posterior']
    data=pd.DataFrame(index=np.arange(len(y)),columns=feature_names+['employment_type'])
    for f in frames:
        data.loc[f['vi'],feature_names+['employment_type']]=f['val'][feature_names+['employment_type']].to_numpy()
    segments={}
    for c in feature_names[:4]:
        x=pd.to_numeric(data[c]);segments[c]=np.select([x.isna(),x<=0,x<=3],['missing','zero','1_to_3'],default='over3')
    segments['employment_type']=data.employment_type.fillna('missing').astype(str).to_numpy()
    if 'decision_datetime' in tr:
        segments['decision_quarter']=pd.to_datetime(tr.decision_datetime).dt.to_period('Q').astype(str).to_numpy()
    segments['baseline_risk_band']=np.select([reference<.05,reference<.15,reference<.4],['under005','005_to015','015_to04'],default='over04')
    segment_rows=[]
    for name,labels in segments.items():
        for value in sorted(set(labels)):
            m=np.asarray(labels)==value;n=int(m.sum());events=int(y[m].sum())
            row=dict(segment=name,value=value,n=n,events=events,event_rate=float(y[m].mean()))
            if n>=200 and min(events,n-events)>=25:
                a,b=scores(y[m],reference[m]),scores(y[m],pred[m]);row.update(reference_KS=a['KS'],KS=b['KS'],delta_KS=b['KS']-a['KS'],reference_AUC=a['AUC'],AUC=b['AUC'],delta_AUC=b['AUC']-a['AUC'])
            else:row['interpretation']='INSUFFICIENT_EVENTS_FOR_STABLE_COMPARISON'
            segment_rows.append(row)
    delta=pred-reference
    # These examples are held-out observations, not training-set explanations.
    case=pd.DataFrame(dict(application_id=tr.application_id,fold=folds,target=y,reference_prediction=reference,prediction=pred,delta_probability=delta))
    cases=pd.concat([case[(y==1)&(reference<.15)].nlargest(30,'delta_probability').assign(case_type='previously_low_scored_bad_moved_up'),case[(y==0)&mask20].nsmallest(30,'delta_probability').assign(case_type='previously_high_scored_good_moved_down'),case[(y==1)&(delta<0)].nsmallest(30,'delta_probability').assign(case_type='bad_moved_down'),case[(y==0)&(delta>0)].nlargest(30,'delta_probability').assign(case_type='good_moved_up')])
    cases.to_csv(out/'ERROR_CASES.csv',index=False)
    pd.DataFrame(segment_rows).to_csv(out/'ERROR_SEGMENTS.csv',index=False)
    report=dict(source='held-out OOF only; descriptive and post-selection, no in-sample predictions',risk_cutoffs=analysis,segments=segment_rows,mean_absolute_probability_change=float(np.mean(abs(delta))),target_direction_verified=bool(scores(y,pred)['AUC']>.5),interpretation='Segment effects are associations and exploratory; do not fit gates on outer labels.')
    dump(report,out/'ERROR_ANALYSIS.json')
    known=[r for r in segment_rows if 'delta_KS' in r]
    text=['# Held-out error analysis','',report['source'],'','| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |','|---|---:|---:|---:|---:|']
    for r in analysis:text.append(f"| {r['top_population_fraction']:.6f} | {r['net_bad_captured']} | {r['new_members']} | {r['entrant_bad_rate']} | {r['exit_bad_rate']} |")
    text+=['','## Segment changes','', '| Segment | Value | N | Events | Delta KS | Delta AUC |','|---|---|---:|---:|---:|---:|']
    for r in sorted(known,key=lambda r:r['delta_KS'],reverse=True):text.append(f"| {r['segment']} | {r['value']} | {r['n']} | {r['events']} | {r['delta_KS']:+.6f} | {r['delta_AUC']:+.6f} |")
    text+=['',report['interpretation'],'','ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.']
    (out/'ERROR_ANALYSIS.md').write_text('\n'.join(text))
    return report


def evaluate(spec,loaded):
    started=time.time();tr,te,sample,y,folds,reference,frames,parent=loaded
    name=spec['id'];out=V/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
    (out/'SOURCE_modeling_features.py').write_bytes(Path(__file__).read_bytes())
    for source in ['modeling_representations.py','modeling_experts.py','modeling_ranking.py']:
        if (V/'scripts'/source).exists():(out/('SOURCE_'+source)).write_bytes((V/'scripts'/source).read_bytes())
    dump(dict(status='RUNNING',spec=spec,pid=os.getpid()),out/'STATUS.json')
    oof=np.full(len(tr),np.nan);pt=np.zeros(len(te));checks=[]
    derived={'cf_share':['modeling_cf_negative_share'],'cf_bayes_burden':['modeling_cf_credible_stress_burden']}.get(spec.get('feature_transform'),[])
    columns=parent['columns']+derived;weight=spec.get('kumo_weight',.75);strength=spec['strength'];blend=spec.get('blend','probability')
    expert=None;expert_audits=None
    if spec.get('expert_key'):
        from modeling_experts import ensure
        expert,expert_audits=ensure(spec['expert_key'],loaded)
    for source_frame in frames:
        f=source_frame
        if derived:
            f=dict(source_frame)
            k=source_frame['h'].risk_conditional_negative_months.to_numpy(float)
            nh=k+source_frame['h'].risk_conditional_nonnegative_months.to_numpy(float)
            mu=float((k.sum()+.5)/(nh.sum()+1.))
            transform_meta=dict(pooled_negative_CF_mean=mu,prior_strength=10.,reference_fit='6k correction-training only')
            for part in ['h','val','test']:
                d=source_frame[part].copy();n=d.risk_conditional_negative_months+d.risk_conditional_nonnegative_months
                if spec['feature_transform']=='cf_share':d[derived[0]]=np.where(n>0,d.risk_conditional_negative_months/np.where(n>0,n,1.),np.nan)
                else:
                    posterior=(d.risk_conditional_negative_months+10.*mu)/(n+10.)
                    d[derived[0]]=np.where(n>0,posterior*d.installment_observed_income_ratio,np.nan)
                f[part]=d
        head=f['head']
        if spec.get('rankloss'):
            from modeling_ranking import fit as fit_ranking
            head=fit_ranking(f,spec)
        elif spec.get('representation'):
            from modeling_representations import fit as fit_representation
            head=fit_representation(f,spec)
        elif spec.get('refit',False):head=fit_head(f['h'],f['kh'],columns,spec['penalty'],strength)
        else:head=dict(head,spec=dict(head['spec'],correction_strength=strength))
        def predict(part,k):
            if spec.get('representation'):
                from modeling_representations import predict as predict_representation
                return predict_representation(head,f[{'correction':'h','validation':'val','test':'test'}[part]],k)
            if not spec.get('refit',False):
                frame=f[{'correction':'h','validation':'val','test':'test'}[part]]
                x,z=f['basis'][part] if part in f['basis'] else design(head,frame,k)
                d=x@head['coefficients'];return expit(z+strength*d),d
            return corrected_probability(head,f[{'correction':'h','validation':'val','test':'test'}[part]],k)
        ph,dh=predict('correction',f['kh']);pv,dv=predict('validation',f['kv']);test,dt=predict('test',f['kt'])
        prior_shift=0.
        if spec.get('prior_alignment'):
            priors=f['training_priors']
            prior_shift=spec.get('prior_alignment_alpha',1.)*(logit(priors['outer_training'])-((1-strength)*logit(priors['context'])+strength*logit(priors['correction'])))
            ph,pv,test=[expit(logit(np.clip(a,1e-6,1-1e-6))+prior_shift) for a in [ph,pv,test]]
        def combine(a,b):
            if blend=='logit':return expit(weight*logit(np.clip(a,1e-6,1-1e-6))+(1-weight)*logit(np.clip(b,1e-6,1-1e-6)))
            return weight*a+(1-weight)*b
        p,tp=combine(pv,f['sv']),combine(test,f['st'])
        if expert is not None:
            w=spec['expert_weight'];p=(1-w)*p+w*expert[f['fold']]['validation'];tp=(1-w)*tp+w*expert[f['fold']]['test']
        oof[f['vi']]=p;pt+=pd.Series(tp,index=f['test'].application_id).reindex(sample.application_id).to_numpy()/5
        bstat=metric(y[f['vi']],reference[f['vi']]);m=metric(y[f['vi']],p)
        row=dict(fold=f['fold'],metrics=m,baseline=bstat,delta_KS=m['KS']-bstat['KS'],delta_AUC=m['AUC']-bstat['AUC'],correction_rows=6000,parameters=len(head['coefficients']),head_training_KS=scores(f['h'].target,ph)['KS'],heldout_corrected_Kumo_KS=scores(f['val'].target,pv)['KS'],train_validation_head_KS_gap=scores(f['h'].target,ph)['KS']-scores(f['val'].target,pv)['KS'],mean_abs_correction=float(np.mean(abs(dv))),P99_abs_correction=float(np.quantile(abs(dv),.99)),optimizer_gradient_max=head['gradient_max_abs'],optimizer_iterations=head['iterations'],train_ids_sha256=ids_hash(tr,f['hi']),validation_ids_sha256=ids_hash(tr,f['vi']))
        if spec.get('prior_alignment'):row.update(prior_alignment_shift=float(prior_shift),training_only_class_priors=f['training_priors'])
        if derived:row['feature_transform_reference']=transform_meta
        checks.append(row);joblib.dump(dict(head=head,kumo_weight=weight,blend=blend,prior_alignment_shift=float(prior_shift),test_averaging='probability_mean_5fold'),out/f"fold{f['fold']}_correction.joblib",compress=3)
        if expert is not None:
            row['additional_expert_audit']=expert_audits[f['fold']]
            dump(dict(expert_key=spec['expert_key'],expert_weight=spec['expert_weight'],checkpoint=f"v7/modeling/experts/{spec['expert_key']}/fold{f['fold']}.joblib"),out/f"fold{f['fold']}_expert_reference.json")
        pd.DataFrame(dict(application_id=f['val'].application_id,prediction=p,correction_logit=dv)).to_parquet(out/f"fold{f['fold']}_validation.parquet",index=False)
        pd.DataFrame(dict(application_id=f['test'].application_id,prediction=tp)).to_parquet(out/f"fold{f['fold']}_test.parquet",index=False)
        dump(dict(status='RUNNING',folds_complete=f['fold']+1,spec=spec,pid=os.getpid()),out/'STATUS.json')
    assert np.isfinite(oof).all() and np.isfinite(pt).all()
    submission=V/'submissions'/f'submission_{name}_V25_modeling.csv'
    pd.DataFrame(dict(application_id=sample.application_id,default_probability=pt)).to_csv(submission,index=False)
    pd.DataFrame(dict(application_id=tr.application_id,target=y,fold=folds,prediction=oof,baseline_prediction=reference)).to_csv(out/'OOF.csv',index=False)
    intervals=bootstrap(y,reference,oof,folds);m,b=metric(y,oof),metric(y,reference)
    error=error_analysis(out,tr,y,folds,reference,oof,frames)
    formula=f"corrected=sigmoid(logit(p_Kumo)+{strength}*X_V25@beta); final="+(f"sigmoid({weight}*logit(corrected)+{1-weight}*logit(p_spline77))" if blend=='logit' else f"{weight}*corrected+{1-weight}*p_spline77")+'; mean5fold test probabilities'
    if spec.get('representation'):
        formula=formula.replace('X_V25@beta','(X_V25@beta + interaction_representation)')
    if expert is not None:
        formula=f"p_final=(1-{spec['expert_weight']})*p_V25+{spec['expert_weight']}*p_{spec['expert_key']}_116; expert trained56k outer-training with inner early stopping; mean5fold test"
    if spec.get('rankloss'):
        formula+='; beta=beta_V25+student_theta/strength; student trained with '+spec['rankloss']+' loss and teacher tether'
    if derived:formula+='; feature_transform='+spec['feature_transform']+' from V25 counts; undefined exposure stays missing; training-only reference'
    if spec.get('prior_alignment'):formula+='; corrected Kumo logit adds alpha*(logit(pi_outertrain)-(1-strength)*logit(pi_context)-strength*logit(pi_correction)); priors from training only'
    result=dict(status='COMPLETE',experiment=name,spec=spec,columns=columns,new_features=parent['new_features'],support_features=parent['support_features'],OOF=m,baseline=b,comparison_baseline='V25',delta_KS=m['KS']-b['KS'],delta_AUC=m['AUC']-b['AUC'],mean_fold_KS=float(np.mean([q['metrics']['KS'] for q in checks])),std_fold_KS=float(np.std([q['metrics']['KS'] for q in checks],ddof=1)),delta_mean_fold_KS=float(np.mean([q['delta_KS'] for q in checks])),fold_wins=sum(q['delta_KS']>0 for q in checks),folds=checks,confidence_interval=intervals,evidence='TARGET_REACHED_DEVELOPMENT_ONLY' if m['KS']>=TARGET else 'INCONCLUSIVE_OR_NEGATIVE',overfitting_assessment=dict(development_folds_reused=True,independent_holdout=False,hyperparameters_prespecified=True,retrained_seeds=1,extra_head_features=0,selection_corrected_CI=False,public_score_unknown=True,risk='HIGH_SEARCH_MULTIPLICITY; do not equate pooled target with hidden-test improvement',diagnostic='Head training metric is optimistic; error analysis uses only outer-held-out predictions.'),frozen_Kumo_fraction=1.,backbone_retrained=False,original_spline_unchanged=True,all_outer_indices_identical=True,point_in_time_certified=False,inherited_baseline_snapshot_uncertainty=True,new_features_DPD_used=parent.get('new_features_DPD_used',False),new_features_snapshot_dependent=False,effective_quarantined=False,submission=str(submission.relative_to(ROOT)),submission_sha256=sha(submission),public_score=None,elapsed_seconds=time.time()-started,source_sha256=sha(__file__),source_feature_manifest='v7/modeling/reports/CACHE_AUDIT.json',feature_matrix_sha256=sha(V/'data/features_raw.parquet'),extension_feature_matrix_sha256=sha(V/'extension/data/raw_features.parquet'),features_changed_vs_V25=False,modeling_formula=formula,modeling_objective=f"mean BCE with unweighted labels + {spec['penalty']}/2*sum(beta[1:]^2); fit full correction, shrink at inference",controller_source='v7/scripts/modeling_features.py',error_analysis=dict(json=str((out/'ERROR_ANALYSIS.json').relative_to(ROOT)),markdown=str((out/'ERROR_ANALYSIS.md').relative_to(ROOT)),cases=str((out/'ERROR_CASES.csv').relative_to(ROOT))))
    if spec.get('representation'):
        result['modeling_objective'] += f" + {spec['interaction_penalty']}/2*||interaction_parameters||^2; representation={spec['representation']}"
        result['representation_source_sha256']=sha(V/'scripts/modeling_representations.py')
        result['representation_rank']=spec.get('rank')
    if expert is not None:
        result['modeling_objective']='Frozen V25 baseline; additional histogram gradient boosting logloss on outer-training56k; tree count selected by inner validation only; fixed blending weight'
        result['expert_audits']=expert_audits;result['expert_source_sha256']=sha(V/'scripts/modeling_experts.py')
    if spec.get('rankloss'):
        result['modeling_objective']=f"BCE(teacher_eta+Xtheta) + {spec['loss_weight']}*{spec['rankloss']}_loss + {spec['teacher_tether']}/2*mean((Xtheta)^2) +.03/2*sum(theta[1:]^2); training rows only"
        result['ranking_source_sha256']=sha(V/'scripts/modeling_ranking.py')
    if derived:
        result['new_features']=parent['new_features']+derived;result['features_changed_vs_V25']=True
        result['derived_feature_formula']='negative eligible-CF months / total eligible-CF months, missing when total0; same V25 inputs/cutoff' if spec['feature_transform']=='cf_share' else '(negative_months+10*mu_training)/(total_months+10)*installment_to_observed_income; NA for total0 or undefined burden; mu from correction rows only'
    dump(result,out/'RESULT.json')
    gap=lambda r:float(np.mean([f['train_validation_head_KS_gap'] for f in r['folds']]))
    control=dict(control='V25',candidate_metrics=m,control_metrics=b,delta_KS=result['delta_KS'],delta_AUC=result['delta_AUC'],KS_fold_wins=result['fold_wins'],delta_mean_fold_KS=result['delta_mean_fold_KS'],per_fold=[dict(fold=q['fold'],delta_KS=q['delta_KS'],delta_AUC=q['delta_AUC']) for q in checks],confidence_interval=intervals,head_gap_candidate=gap(result),head_gap_control=gap(parent),head_gap_delta=gap(result)-gap(parent),parameter_count_candidate=[f['parameters'] for f in checks],parameter_count_control=[f['parameters'] for f in parent['folds']])
    dump(control,out/'MATCHED_CONTROL.json');dump(dict(status='COMPLETE',submission_sha256=result['submission_sha256']),out/'STATUS.json')
    report=f"# {name}: {spec['label']}\n\n{formula}\n\nOOF KS={m['KS']:.8f}; delta vsV25={result['delta_KS']:+.8f}; AUC={m['AUC']:.8f}; deltaAUC={result['delta_AUC']:+.8f}. KS wins {result['fold_wins']}/5.95% deltaKS CI={intervals['delta_KS']}.\n\nCorrection train-validation KS gap={control['head_gap_candidate']:.8f} versusV25 {control['head_gap_control']:.8f}.\n\n{intervals['caveat']}\n\nError analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.\n\nSubmission `{result['submission']}` SHA256 `{result['submission_sha256']}`.\n"
    (out/'REPORT.md').write_text(report);dump(dict(method=result['modeling_formula'],objective=result['modeling_objective'],spec=spec,source=result['controller_source'],columns=columns,metrics=m,error_analysis=result['error_analysis'],source_sha256=result['source_sha256']),out/'MODELING_METHOD.json')
    print('MODEL_COMPLETE',name,json.dumps(dict(KS=m['KS'],AUC=m['AUC'],deltaKS=result['delta_KS'],wins=result['fold_wins'],target_reached=m['KS']>=TARGET)),flush=True)
    return result


def main(requested=None):
    (V/'modeling/reports').mkdir(parents=True,exist_ok=True)
    guard=(V/'modeling.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    all_specs=json.loads((V/'modeling_plan.json').read_text())
    pending=[s for s in all_specs if not (V/'runs'/s['id']/'RESULT.json').exists()]
    specs=[s for s in pending if s['id']==requested] if requested else pending[:1]
    assert len(specs)==1,'Run exactly one new experiment, then review and push before the next'
    gate=json.loads((V/'modeling/reports/REPO_REVIEW_GATE.json').read_text())
    assert gate['experiment']==specs[0]['id'] and gate['status']=='REVIEWED_FOR_NEXT_RUN'
    request=urllib.request.Request('https://api.github.com/repos/TaqiyudinMiftah/Logika/git/ref/heads/main',headers={'User-Agent':'Logika-experiment-review'})
    with urllib.request.urlopen(request,timeout=20) as response:remote=json.load(response)
    assert remote['object']['sha']==gate['reviewed_head'],'Peer pushed new work; review before training'
    cg=Path('/sys/fs/cgroup');headroom=int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())
    status={s.split(':',1)[0]:s.split(':',1)[1].strip() for s in Path('/proc/self/status').read_text().splitlines() if ':' in s}
    resident=int(status['VmRSS'].split()[0])*1024
    needed=max(0,450*2**20-resident)+150*2**20
    dump(dict(headroom_bytes=headroom,resident_bytes=resident,projected_remaining_plus_reserve_bytes=needed,streaming_prediction_bases=True),V/'modeling/reports/LAUNCH_MEMORY.json')
    assert headroom>=needed,'Insufficient RAM headroom for streamed cached-model process plus reserve'
    state('LOADING_FIXED_V25_CACHE',experiments=[s['id'] for s in specs])
    loaded=load()
    baseline_out=V/'modeling/reports';error_analysis(baseline_out,loaded[0],loaded[3],loaded[4],loaded[5],loaded[5],loaded[6])
    from master_log import generate
    done=[];results=[]
    for spec in specs:
        path=V/'runs'/spec['id']/'RESULT.json'
        if path.exists():result=json.loads(path.read_text())
        else:
            state('RUNNING',experiment=spec['id'],complete=done)
            result=evaluate(spec,loaded)
        results.append(result);done.append(spec['id'])
        generate()
        best=max(results,key=lambda r:r['OOF']['KS'])
        state('RUNNING',experiment=spec['id'],complete=done,best_variant=best['experiment'],best_metrics=best['OOF'],target_reached=best['OOF']['KS']>=TARGET)
    results=[json.loads((V/'runs'/s['id']/'RESULT.json').read_text()) for s in all_specs if (V/'runs'/s['id']/'RESULT.json').exists()]
    best=max(results,key=lambda r:r['OOF']['KS'])
    table=pd.DataFrame([dict(experiment=r['experiment'],method=r['spec']['label'],KS=r['OOF']['KS'],AUC=r['OOF']['AUC'],delta_KS=r['delta_KS'],delta_AUC=r['delta_AUC'],KS_fold_wins=r['fold_wins'],head_gap=float(np.mean([f['train_validation_head_KS_gap'] for f in r['folds']])),CI_low=r['confidence_interval']['delta_KS'][0],CI_high=r['confidence_interval']['delta_KS'][1],submission=r['submission']) for r in results])
    table.to_csv(V/'modeling/reports/MODELING_EXPERIMENTS.csv',index=False)
    state('COMPLETE',complete=done,best_variant=best['experiment'],best_metrics=best['OOF'],target_reached=best['OOF']['KS']>=TARGET)
    generate()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run');args=parser.parse_args()
    try:main(args.run)
    except Exception:
        detail=traceback.format_exc();state('FAILED',traceback=detail);raise
