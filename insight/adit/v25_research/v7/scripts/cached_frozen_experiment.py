"""Exact queued V172/V177 recipes from completed frozen-model caches.

No new model fit is needed for a fixed coverage gate or fixed score average.
This CPU implementation preserves the planned formulas and validation folds.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
from pathlib import Path
import argparse,json,sys,time,shutil,urllib.request
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import dump,sha,scores,ids_hash
from experiments import bootstrap
from modeling_features import error_analysis

def main(ident):
    assert ident in ['V172','V177'];started=time.time();out=V/'runs'/ident;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
    spec=next(s for s in json.loads((V/'kumo_plan.json').read_text()) if s['id']==ident)
    gate=json.loads((V/'modeling/reports/REPO_REVIEW_GATE.json').read_text());assert gate['experiment']==ident
    req=urllib.request.Request('https://api.github.com/repos/TaqiyudinMiftah/Logika/git/ref/heads/main',headers={'User-Agent':'Logika-research'})
    with urllib.request.urlopen(req,timeout=20) as r:remote=json.load(r)
    assert remote['object']['sha']==gate['reviewed_head'],'Peer updates require review before this experiment'
    dump(dict(status='REPLAYING_FROZEN_CHECKPOINT_PREDICTIONS',experiment=ident,pid=os.getpid()),out/'STATUS.json')
    parent=json.loads((V/'runs/V25/RESULT.json').read_text());source_id='V171' if ident=='V172' else 'V169'
    source=json.loads((V/'runs'/source_id/'RESULT.json').read_text())
    assert source['frozen_Kumo_fraction']==1. and source['all_outer_indices_identical']
    fields=['application_id','target','employment_type','installment_observed_income_ratio','re_late_30_posterior']
    schema=pq.read_schema(ROOT/'data/train_features.parquet').names
    if 'decision_datetime' in schema:fields+=['decision_datetime']
    tr=pd.read_parquet(ROOT/'data/train_features.parquet',columns=fields)
    support=parent['support_features'];additional=parent['new_features']
    raw=pd.read_parquet(V/'data/features_raw.parquet',columns=['application_id']+support)
    counts=pd.read_parquet(V/'extension/data/raw_features.parquet',columns=['application_id','ext_aux_allocation_negative_n','ext_aux_allocation_nonnegative_n']).rename(columns={'ext_aux_allocation_negative_n':additional[0],'ext_aux_allocation_nonnegative_n':additional[1]})
    raw=raw.merge(counts,on='application_id',validate='one_to_one');tr=tr.merge(raw,on='application_id',validate='one_to_one')
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    reference=pd.read_csv(V/'runs/V25/OOF.csv');assert reference.application_id.tolist()==tr.application_id.tolist()
    assert np.array_equal(reference.target.to_numpy(),tr.target.to_numpy())
    y=tr.target.to_numpy(int);baseline=reference.prediction.to_numpy();folds=np.full(len(tr),-1);oof=np.full(len(tr),np.nan);test_pred=np.zeros(len(sample));checks=[];frames=[]
    for fold in range(5):
        a=ROOT/'artifacts'/f'fold{fold}';idx=np.load(a/'indices.npz');vi,hi,ci=[idx[k] for k in ['validation','correction','context']]
        assert (len(vi),len(hi),len(ci))==(14000,6000,50000) and len(set(np.r_[vi,hi,ci]))==70000
        folds[vi]=fold;preds={}
        for part in ['validation','test']:
            anchor=pd.read_parquet(V/'runs/V25'/f'fold{fold}_{part}.parquet');other=pd.read_parquet(V/'runs'/source_id/f'fold{fold}_{part}.parquet')
            assert anchor.application_id.tolist()==other.application_id.tolist()
            p0=anchor.prediction.to_numpy();p1=other.prediction.to_numpy()
            if part=='validation':assert anchor.application_id.tolist()==tr.application_id.iloc[vi].tolist()
            else:assert set(anchor.application_id)==set(sample.application_id)
            if ident=='V172':
                observed=raw.set_index('application_id').reindex(anchor.application_id).aux_allocation_count.to_numpy(float)>0
                p=np.where(observed,p1,p0)
            else:
                assert spec['score_blend_weight']==.1;p=.9*p0+.1*p1
            assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            output=pd.DataFrame(dict(application_id=anchor.application_id,prediction=p));output.to_parquet(out/f'fold{fold}_{part}.parquet',index=False);preds[part]=p
            if part=='validation':oof[vi]=p
            else:test_pred+=pd.Series(p,index=anchor.application_id).reindex(sample.application_id).to_numpy()/5
        sm=source['folds'][fold];m=scores(y[vi],preds['validation']);b=scores(y[vi],baseline[vi])
        row=dict(fold=fold,metrics=m,baseline=b,delta_KS=m['KS']-b['KS'],delta_AUC=m['AUC']-b['AUC'],parameters=sm['parameters'],correction_rows=6000,head_training_KS=sm['head_training_KS'],heldout_corrected_Kumo_KS=sm['heldout_corrected_Kumo_KS'],train_validation_head_KS_gap=sm['train_validation_head_KS_gap'],train_ids_sha256=ids_hash(tr,hi),validation_ids_sha256=ids_hash(tr,vi),context_ids_sha256=ids_hash(tr,ci),diagnostic='Head gap belongs to frozen source head; no gate/meta-model was trained on these rows',source_prediction_sha256=sha(V/'runs'/source_id/f'fold{fold}_validation.parquet'))
        checks.append(row);frames.append(dict(vi=vi,val=tr.iloc[vi]));shutil.copyfile(V/'runs'/source_id/f'fold{fold}_correction.joblib',out/f'fold{fold}_correction.joblib')
        dump(dict(frozen_model_sources=['V25',source_id],source_checkpoints=[f'v7/runs/{s}/fold{fold}_correction.joblib' for s in ['V25',source_id]],source_predictions={s:{p:sha(V/'runs'/s/f'fold{fold}_{p}.parquet') for p in ['validation','test']} for s in ['V25',source_id]},recipe=spec,backbone_frozen_fraction=1.,new_training=False),out/f'fold{fold}_checkpoint.json')
    assert np.isfinite(oof).all() and np.isfinite(test_pred).all()
    submission=V/'submissions'/f'submission_{ident}_Kumo_Large_V25.csv'
    pd.DataFrame(dict(application_id=sample.application_id,default_probability=test_pred)).to_csv(submission,index=False)
    pd.DataFrame(dict(application_id=tr.application_id,target=y,fold=folds,prediction=oof,baseline_prediction=baseline)).to_csv(out/'OOF.csv',index=False)
    m,b=scores(y,oof),scores(y,baseline);interval=bootstrap(y,baseline,oof,folds)
    error_analysis(out,tr,y,folds,baseline,oof,frames)
    formula='p_final=where(aux_allocation_count>0,p_V171,p_V25); exact completed source checkpoints' if ident=='V172' else 'p_final=.90*p_V25+.10*p_V169; fixed matched-fold score average'
    caveat='Development-selected recipe; conditional bootstrap excludes search/retraining uncertainty. Source-head gap is diagnostic only; no new supervised gate or meta-weights fitted.'
    result=dict(status='COMPLETE',experiment=ident,spec=spec,columns=parent['columns'],new_features=parent['new_features'],support_features=support,OOF=m,baseline=b,comparison_baseline='V25',delta_KS=m['KS']-b['KS'],delta_AUC=m['AUC']-b['AUC'],mean_fold_KS=float(np.mean([x['metrics']['KS'] for x in checks])),std_fold_KS=float(np.std([x['metrics']['KS'] for x in checks],ddof=1)),delta_mean_fold_KS=float(np.mean([x['delta_KS'] for x in checks])),fold_wins=sum(x['delta_KS']>0 for x in checks),folds=checks,confidence_interval=interval,submission=str(submission.relative_to(ROOT)),submission_sha256=sha(submission),public_score=None,elapsed_seconds=time.time()-started,source_sha256=sha(__file__),controller_source='v7/scripts/cached_frozen_experiment.py',modeling_formula=formula,modeling_objective='No additional fitting; replay fixed source predictions and apply prespecified gate/weights',frozen_Kumo_fraction=1.,backbone_retrained=False,backbone_gradient_updated=False,in_context_refitted=False,original_spline_unchanged=True,all_outer_indices_identical=True,point_in_time_certified=False,inherited_baseline_snapshot_uncertainty=True,features_changed_vs_V25=False,source_results={s:sha(V/'runs'/s/'RESULT.json') for s in ['V25',source_id]},overfitting_assessment=dict(development_folds_reused=True,independent_holdout=False,new_supervised_parameters=0,selection_corrected_CI=False,public_score_unknown=True,risk='HIGH_SEARCH_MULTIPLICITY',diagnostic=caveat),error_analysis=dict(json=str((out/'ERROR_ANALYSIS.json').relative_to(ROOT)),markdown=str((out/'ERROR_ANALYSIS.md').relative_to(ROOT)),cases=str((out/'ERROR_CASES.csv').relative_to(ROOT))))
    result['features_changed_vs_V25']=ident=='V172'
    result['context_columns']=source.get('context_columns')
    result['derived_context_features']=source.get('derived_context_features',[])
    if ident=='V172':result['new_features']+=result['derived_context_features']
    dump(dict(method=formula,spec=spec,source_prediction_controls=result['source_results'],source_sha256=sha(__file__),new_training=False),out/'MODELING_METHOD.json')
    dump(dict(control='V25',candidate_metrics=m,control_metrics=b,delta_KS=result['delta_KS'],delta_AUC=result['delta_AUC'],confidence_interval=interval,head_gap_candidate=float(np.mean([f['train_validation_head_KS_gap'] for f in checks])),head_gap_control=float(np.mean([f['train_validation_head_KS_gap'] for f in parent['folds']])),head_gap_caveat='Source head only; no new head fitting'),out/'MATCHED_CONTROL.json')
    (out/'REPORT.md').write_text(f"# {ident}: {spec['label']}\n\n{formula}\n\nKS={m['KS']:.10f}; AUC={m['AUC']:.10f}; deltaKS={result['delta_KS']:+.10f}; paired95% CI={interval['delta_KS']}.\n\n{caveat}\n\nThis CPU cache replay evaluates the exact queued prediction recipe without unnecessary Kumo/GPU refitting. All5folds, ID alignment and frozen-source checks verified. Public score unknown; V25 remains incumbent.\n")
    dump(result,out/'RESULT.json');dump(dict(status='COMPLETE',experiment=ident,metrics=m,target_reached=m['KS']>=.531),out/'STATUS.json')
    from master_log import generate
    generate();print('COMPLETE',ident,json.dumps(m),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    try:main(args.run)
    except Exception:
        import traceback
        dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json');raise
