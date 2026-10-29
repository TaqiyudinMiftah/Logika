"""Check completed frozen-Kumo results against actual labels, folds and files."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path.insert(0,str(ROOT/'scripts'))
from baseline import scores,sha,ids_hash,dump

def audit(ident):
    out=V/'runs'/ident;r=json.loads((out/'RESULT.json').read_text())
    assert r['status']=='COMPLETE' and r['experiment']==ident
    assert r['frozen_Kumo_fraction']==1. and not r.get('backbone_gradient_updated',False)
    assert r['all_outer_indices_identical'] and len(r['folds'])==5
    train=pd.read_parquet(ROOT/'data/train_features.parquet',columns=['application_id','customer_id','target'])
    test=pd.read_parquet(ROOT/'data/test_features.parquet',columns=['application_id','customer_id'])
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    assert (len(train),len(test))==(70000,30000)
    assert train.application_id.is_unique and train.customer_id.is_unique and test.application_id.is_unique and test.customer_id.is_unique
    assert set(train.application_id).isdisjoint(test.application_id) and set(train.customer_id).isdisjoint(test.customer_id)
    oof=pd.read_csv(out/'OOF.csv',float_precision='round_trip');assert oof.application_id.tolist()==train.application_id.tolist()
    assert np.array_equal(oof.target.to_numpy(),train.target.to_numpy()) and np.isfinite(oof.prediction).all()
    assert np.all((oof.prediction>=0)&(oof.prediction<=1))
    measured=scores(train.target.to_numpy(),oof.prediction.to_numpy())
    for key in ['KS','AUC']:assert abs(measured[key]-r['OOF'][key])<1e-12,(ident,key,measured,r['OOF'])
    assert measured['AUC']>.5 and measured['signed_KS']>0,'High score must mean higher default risk'
    baseline=pd.read_csv(V/'runs/V25/OOF.csv',float_precision='round_trip');assert baseline.application_id.tolist()==train.application_id.tolist()
    base=scores(train.target.to_numpy(),baseline.prediction.to_numpy())
    assert abs(r['delta_KS']-(measured['KS']-base['KS']))<1e-12
    assert abs(r['delta_AUC']-(measured['AUC']-base['AUC']))<1e-12
    if 'comparison_to_V185' in r:
        p185=pd.read_csv(V/'runs/V185/OOF.csv',float_precision='round_trip');assert p185.application_id.tolist()==train.application_id.tolist()
        m185=scores(train.target.to_numpy(),p185.prediction.to_numpy());compare=r['comparison_to_V185']
        for key in ['KS','AUC']:assert abs(compare['delta_'+key]-(measured[key]-m185[key]))<1e-12
        assert compare['confidence_interval']['repetitions']>=200 and (out/'GENERALIZATION_REVIEW.json').exists()
    if 'comparison_to_V193' in r:
        p193=pd.read_csv(V/'runs/V193/OOF.csv',float_precision='round_trip');assert p193.application_id.tolist()==train.application_id.tolist()
        m193=scores(train.target.to_numpy(),p193.prediction.to_numpy());compare=r['comparison_to_V193']
        for key in ['KS','AUC']:assert abs(compare['delta_'+key]-(measured[key]-m193[key]))<1e-12
        assert compare['confidence_interval']['repetitions']>=200 and (out/'CONTINUOUS_INSIGHT.json').exists()
    assigned=np.zeros(len(train),dtype=int);test_mean=np.zeros(len(test));fold_audits=[]
    for fold in range(5):
        index=np.load(ROOT/'artifacts'/f'fold{fold}/indices.npz');ci,hi,vi=[index[k] for k in ['context','correction','validation']]
        assert (len(ci),len(hi),len(vi))==(50000,6000,14000) and len(set(np.r_[ci,hi,vi]))==70000
        assigned[vi]+=1;assert np.all(oof.fold.to_numpy()[vi]==fold)
        f=next(x for x in r['folds'] if x['fold']==fold)
        assert ids_hash(train,hi)==f['train_ids_sha256'] and ids_hash(train,vi)==f['validation_ids_sha256']
        if 'new_spline_training_rows' in f:
            assert f['new_spline_training_rows']==56000
            assert ids_hash(train,np.r_[ci,hi])==f['new_spline_training_ids_sha256']
        if 'new_expert_training_rows' in f:
            assert f['new_expert_training_rows']==56000
            assert ids_hash(train,np.r_[ci,hi])==f['new_expert_training_ids_sha256']
        if 'context_ids_sha256' in f:assert ids_hash(train,ci)==f['context_ids_sha256']
        val=pd.read_parquet(out/f'fold{fold}_validation.parquet');te=pd.read_parquet(out/f'fold{fold}_test.parquet')
        assert val.application_id.tolist()==train.application_id.iloc[vi].tolist()
        assert te.application_id.is_unique and set(te.application_id)==set(sample.application_id)
        assert np.max(abs(val.prediction.to_numpy()-oof.prediction.to_numpy()[vi]))<1e-14
        metric=scores(train.target.to_numpy()[vi],val.prediction.to_numpy())
        for key in ['KS','AUC']:assert abs(metric[key]-f['metrics'][key])<1e-12
        test_mean+=pd.Series(te.prediction.to_numpy(),index=te.application_id).reindex(sample.application_id).to_numpy()/5
        cp=out/f'fold{fold}_correction.joblib';assert cp.exists() and cp.stat().st_size>0
        contexts=[]
        for p in sorted(out.glob(f'fold{fold}_*context_indices.npy')):
            actual=np.load(p);assert np.isin(actual,ci).all() and len(np.unique(actual))==len(actual)
            assert not np.intersect1d(actual,np.r_[hi,vi]).size
            contexts.append(dict(file=p.name,rows=len(actual),sha256=sha(p)))
        fold_audits.append(dict(fold=fold,metrics=metric,context_disjoint=True,correction_disjoint=True,checkpoint_sha256=sha(cp),context_samples=contexts))
    assert np.all(assigned==1),'Each labeled application must have exactly one held-out prediction'
    sub=ROOT/r['submission'];assert sha(sub)==r['submission_sha256']
    submitted=pd.read_csv(sub,float_precision='round_trip')
    assert submitted.columns.tolist()==['application_id','default_probability'] and submitted.application_id.tolist()==sample.application_id.tolist()
    assert np.isfinite(submitted.default_probability).all() and np.all((submitted.default_probability>=0)&(submitted.default_probability<=1))
    error=float(np.max(abs(test_mean-submitted.default_probability.to_numpy())));assert error<1e-14
    required=['REPORT.md','MODELING_METHOD.json','MATCHED_CONTROL.json','ERROR_ANALYSIS.json','ERROR_ANALYSIS.md','ERROR_SEGMENTS.csv','ERROR_CASES.csv']
    assert all((out/p).exists() and (out/p).stat().st_size>0 for p in required)
    assert r['confidence_interval']['repetitions']>=200 and len(r['confidence_interval']['delta_KS'])==2
    assert r['overfitting_assessment']['public_score_unknown'] and r.get('public_score') is None
    feature_names=r.get('context_columns') or []
    assert not set(feature_names)&{'application_id','customer_id','loan_id','target','application_datetime','decision_datetime'}
    value=dict(status='PASS',experiment=ident,UTC=time.time(),measured_OOF=measured,delta_KS=r['delta_KS'],delta_AUC=r['delta_AUC'],labels_and_ids_match=True,all_five_outer_folds_match=True,entity_disjoint=True,frozen_backbone_record=True,score_direction_verified=True,test_rows=30000,submission_sha256=sha(sub),test_mean_max_abs_error=error,required_artifacts=required,folds=fold_audits,goal_threshold_reached=measured['KS']>=.531,limitations='Artifact/metric/split checks do not establish unknown legacy snapshot provenance or independent generalization. CI remains conditional on fitted predictions and repeated development search.',source_sha256=sha(__file__))
    dump(value,out/'REQUIREMENTS_AUDIT.json');print(json.dumps(dict(experiment=ident,status='PASS',KS=measured['KS'],AUC=measured['AUC'],goal_threshold_reached=value['goal_threshold_reached'])),flush=True)
    from update_development_incumbents import update
    update()
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',nargs='+');args=p.parse_args()
    for ident in args.experiment:audit(ident)
