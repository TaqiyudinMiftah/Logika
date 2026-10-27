"""Cross-platform fixed-score fusion on authenticated mirrored source artifacts."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,ast,hashlib,importlib.metadata,json,subprocess,sys,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import scores,sha,ids_hash,dump
from experiments import bootstrap

def save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json.dumps(value,indent=2,allow_nan=False).encode())

def reports():
    # Reuse only platform-independent reporting functions. The source module's
    # Linux process lock is not needed to calculate held-out error diagnostics.
    tree=ast.parse((V/'scripts/modeling_features.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['top_set','error_analysis']]
    assert len(nodes)==2
    scope=dict(np=np,pd=pd,json=json,scores=scores,dump=save)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_heldout_error_reporting','exec'),scope)
    return scope['error_analysis']

def main(ident,expected):
    started=time.time();out=V/'runs'/ident;out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists()
    checkout=ROOT.parent/'shared_repo_sync_20261027'
    subprocess.run(['git','-C',str(checkout),'fetch','origin','main'],check=True,capture_output=True)
    actual=subprocess.check_output(['git','-C',str(checkout),'rev-parse','origin/main'],text=True).strip();assert actual==expected,'Review changed shared HEAD before evaluating a new recipe'
    spec=next(s for s in json.loads((V/'score_fusion_plan.json').read_text()) if s['id']==ident)
    parent=json.loads((V/'runs/V25/RESULT.json').read_text());source_ids=['V25','V167','V178'];source_results={s:json.loads((V/'runs'/s/'RESULT.json').read_text()) for s in source_ids}
    for s,r in source_results.items():assert r['frozen_Kumo_fraction']==1. and r['all_outer_indices_identical']
    fields=['application_id','target','employment_type','installment_observed_income_ratio','re_late_30_posterior']
    if 'decision_datetime' in pq.read_schema(ROOT/'data/train_features.parquet').names:fields+=['decision_datetime']
    tr=pd.read_parquet(ROOT/'data/train_features.parquet',columns=fields);sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    reference=pd.read_csv(V/'runs/V25/OOF.csv',float_precision='round_trip');assert reference.application_id.tolist()==tr.application_id.tolist();assert np.array_equal(reference.target.to_numpy(),tr.target.to_numpy())
    y=tr.target.to_numpy(int);baseline=reference.prediction.to_numpy();base_metric=scores(y,baseline)
    for key in ['KS','AUC']:assert abs(base_metric[key]-parent['OOF'][key])<1e-12,'Local metric implementation must reproduce source control'
    for source in ['V167','V178']:
        saved=pd.read_csv(V/'runs'/source/'OOF.csv',float_precision='round_trip');assert saved.application_id.tolist()==tr.application_id.tolist();m=scores(y,saved.prediction.to_numpy())
        for key in ['KS','AUC']:assert abs(m[key]-source_results[source]['OOF'][key])<1e-12
    oof=np.full(len(tr),np.nan);folds=np.full(len(tr),-1);pt=np.zeros(len(sample));rows=[];frames=[];logit_residuals={s:np.empty(len(tr)) for s in ['V167','V178']}
    def fusion(values):
        if spec['method']=='probability_average':return .5*values['V167']+.5*values['V178']
        z0=logit(np.clip(values['V25'],1e-6,1-1e-6));d=sum(logit(np.clip(values[s],1e-6,1-1e-6))-z0 for s in ['V167','V178'])
        return expit(z0+np.clip(d,-spec['residual_cap'],spec['residual_cap']))
    for fold in range(5):
        idx=np.load(ROOT/'artifacts'/f'fold{fold}/indices.npz');ci,hi,vi=[idx[k] for k in ['context','correction','validation']]
        assert (len(ci),len(hi),len(vi))==(50000,6000,14000) and len(set(np.r_[ci,hi,vi]))==70000;folds[vi]=fold
        source_hashes={};prediction={}
        for part in ['validation','test']:
            values={};ids=None
            for source in source_ids:
                p=V/'runs'/source/f'fold{fold}_{part}.parquet';d=pd.read_parquet(p)
                if ids is None:ids=d.application_id.tolist()
                assert ids==d.application_id.tolist();values[source]=d.prediction.to_numpy();source_hashes[source+'_'+part]=sha(p)
            if part=='validation':assert ids==tr.application_id.iloc[vi].tolist()
            else:assert len(ids)==len(sample) and set(ids)==set(sample.application_id)
            pred=fusion(values);assert np.isfinite(pred).all() and np.all((pred>=0)&(pred<=1))
            pd.DataFrame(dict(application_id=ids,prediction=pred)).to_parquet(out/f'fold{fold}_{part}.parquet',index=False);prediction[part]=pred
            if part=='validation':
                oof[vi]=pred
                z0=logit(np.clip(values['V25'],1e-6,1-1e-6))
                for source in ['V167','V178']:logit_residuals[source][vi]=logit(np.clip(values[source],1e-6,1-1e-6))-z0
            else:pt+=pd.Series(pred,index=ids).reindex(sample.application_id).to_numpy()/5
        base_fold=parent['folds'][fold];m=scores(y[vi],prediction['validation']);b=scores(y[vi],baseline[vi])
        row=dict(fold=fold,metrics=m,baseline=b,delta_KS=m['KS']-b['KS'],delta_AUC=m['AUC']-b['AUC'],parameters=0,new_trainable_parameters=0,correction_rows=6000,head_training_KS=base_fold['head_training_KS'],heldout_corrected_Kumo_KS=base_fold['heldout_corrected_Kumo_KS'],train_validation_head_KS_gap=base_fold['train_validation_head_KS_gap'],head_gap_caveat='Original V25 head diagnostic only; fixed fusion has no newly trained head',train_ids_sha256=ids_hash(tr,hi),validation_ids_sha256=ids_hash(tr,vi),context_ids_sha256=ids_hash(tr,ci),source_prediction_sha256=source_hashes)
        rows.append(row)
        joblib.dump(dict(kind='fixed_score_fusion',recipe=spec,source_checkpoint_references={s:f'v7/runs/{s}/fold{fold}_correction.joblib' for s in source_ids},source_prediction_sha256=source_hashes,new_fitted_parameters=0),out/f'fold{fold}_correction.joblib',compress=3)
        ext=pd.read_parquet(V/'extension/data'/f'fold{fold}/validation.parquet',columns=['application_id']+parent['support_features']+parent['new_features']);assert ext.application_id.tolist()==tr.application_id.iloc[vi].tolist()
        frame=tr.iloc[vi].merge(ext,on='application_id',validate='one_to_one');frames.append(dict(vi=vi,val=frame))
        save(dict(status='FOLD_COMPLETE',execution_host='local_windows',pid=os.getpid(),fold=fold,metrics=m),out/'STATUS.json')
    assert np.isfinite(oof).all() and np.isfinite(pt).all()
    m=scores(y,oof);interval=bootstrap(y,baseline,oof,folds)
    submission=V/'submissions'/f'submission_{ident}_V25_spectral_local_fusion.csv';pd.DataFrame(dict(application_id=sample.application_id,default_probability=pt)).to_csv(submission,index=False,lineterminator='\n')
    pd.DataFrame(dict(application_id=tr.application_id,target=y,fold=folds,prediction=oof,baseline_prediction=baseline)).to_csv(out/'OOF.csv',index=False,lineterminator='\n')
    reports()(out,tr,y,folds,baseline,oof,frames)
    formula=spec['formula'];correlation=float(np.corrcoef(logit_residuals['V167'],logit_residuals['V178'])[0,1])
    result=dict(status='COMPLETE',experiment=ident,spec=spec,columns=parent['columns'],context_columns=json.loads((ROOT/'BASELINE_LOCK.json').read_text())['Kumo']['features'],new_features=parent['new_features'],support_features=parent['support_features'],OOF=m,baseline=base_metric,comparison_baseline='V25',delta_KS=m['KS']-base_metric['KS'],delta_AUC=m['AUC']-base_metric['AUC'],mean_fold_KS=float(np.mean([r['metrics']['KS'] for r in rows])),std_fold_KS=float(np.std([r['metrics']['KS'] for r in rows],ddof=1)),delta_mean_fold_KS=float(np.mean([r['delta_KS'] for r in rows])),fold_wins=sum(r['delta_KS']>0 for r in rows),folds=rows,confidence_interval=interval,frozen_Kumo_fraction=1.,backbone_retrained=False,backbone_gradient_updated=False,in_context_refitted=False,original_spline_unchanged=True,all_outer_indices_identical=True,point_in_time_certified=False,inherited_baseline_snapshot_uncertainty=True,features_changed_vs_V25=True,submission=submission.relative_to(ROOT).as_posix(),submission_sha256=sha(submission),public_score=None,execution_host='local_windows',source_data_host='node480',packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow','joblib']},source_results_sha256={s:sha(V/'runs'/s/'RESULT.json') for s in source_ids},source_logit_residual_correlation=correlation,modeling_formula=formula,modeling_objective='Fixed expert composition, no fitted weights, reference distributions or label-based feature selection in this experiment',controller_source='v7/scripts/score_fusion_local.py',source_sha256=sha(__file__),elapsed_seconds=time.time()-started,overfitting_assessment=dict(development_folds_reused=True,independent_holdout=False,hyperparameters_prespecified=True,new_supervised_parameters=0,selection_corrected_CI=False,public_score_unknown=True,risk='HIGH_SEARCH_MULTIPLICITY',diagnostic='Source choice follows earlier development results. Error correlation does not establish independent evidence. Bootstrap excludes search/retraining uncertainty.'),error_analysis=dict(json=(out/'ERROR_ANALYSIS.json').relative_to(ROOT).as_posix(),markdown=(out/'ERROR_ANALYSIS.md').relative_to(ROOT).as_posix(),cases=(out/'ERROR_CASES.csv').relative_to(ROOT).as_posix()))
    save(dict(method=formula,spec=spec,source_results_sha256=result['source_results_sha256'],source_sha256=sha(__file__),execution_host='local_windows'),out/'MODELING_METHOD.json')
    save(dict(control='V25',candidate_metrics=m,control_metrics=base_metric,delta_KS=result['delta_KS'],delta_AUC=result['delta_AUC'],confidence_interval=interval,head_gap_candidate=float(np.mean([r['train_validation_head_KS_gap'] for r in rows])),head_gap_control=float(np.mean([r['train_validation_head_KS_gap'] for r in rows])),head_gap_caveat='Original V25 head only; no new fitting'),out/'MATCHED_CONTROL.json')
    (out/'REPORT.md').write_bytes(f"# {ident}: {spec['label']}\n\n{formula}\n\nKS={m['KS']:.10f}; AUC={m['AUC']:.10f}; deltaKS={result['delta_KS']:+.10f}; paired95% CI={interval['delta_KS']}.\n\nSource-residual correlation={correlation:.6f}. No source independence is assumed. All5folds and test IDs match original artifacts. Computed on local CPU from authenticated mirrored artifacts, with local package versions recorded; baseline metrics reproduced before evaluation. No Kumo updates or hidden-test labels. Public score unknown; retain V25 incumbent.\n".encode())
    save(result,out/'RESULT.json');save(dict(status='COMPLETE',experiment=ident,metrics=m,goal_threshold_reached=m['KS']>=.531,execution_host='local_windows'),out/'STATUS.json')
    print(json.dumps(dict(experiment=ident,KS=m['KS'],AUC=m['AUC'],delta_KS=result['delta_KS'],CI=interval['delta_KS'],goal_threshold_reached=m['KS']>=.531)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--expected-head',required=True);args=p.parse_args();main(args.run,args.expected_head)
