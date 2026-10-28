"""Validate all declared feature inputs and the bounded fivefold pilots."""
from pathlib import Path
import collections, hashlib, json, sys, time
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
from scipy.stats import ks_2samp
from features import Features, formula_gate

R=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def score(y,p):
    f,t,_=roc_curve(y,p,drop_intermediate=False)
    return {'KS':float(np.max(abs(t-f))),'AUC':float(roc_auc_score(y,p))}

def main():
    started=time.time();plan=json.loads((R/'PLAN.json').read_text());catalog=json.loads((R/'CATALOG.json').read_text())
    assert sha(R/'CATALOG.json')==plan['catalog_sha256']
    assert not (R/'PILOT_GATE.json').exists()
    unit_checks=formula_gate()
    for y,p,want in [([0,0,1,1],[0,1,2,3],1.),([0,1,0,1],[.5,.5,.5,.5],0.),([0,0,1,1],[0,1,1,2],.5)]:
        assert abs(score(y,p)['KS']-want)<1e-12
    # Materialize the complete catalog in small blocks; label-free execution checks.
    f=Features(0);dropped={};finite=[];covered=set()
    for pack in catalog['packs']:
        frame,omit=f.pack(pack)
        covered.update(frame.columns)
        dropped[pack['id']]=omit
        finite.append({'pack':pack['id'],'columns':len(frame.columns),'dropped':len(omit)})
    original=np.load('/mnt/home/faris/logika-faris-exploration/transfer-20261025/ROBUSTNESS_OOF.npz')
    y,folds=original['target'],original['fold'];results={};seconds=[];supports=[]
    for fold in range(5):
        ix=np.load(R/f'packets/fold{fold}/roles.npz')
        assert np.array_equal(y[ix['context']],ix['y_context'])
        assert np.array_equal(y[ix['correction']],ix['y_correction'])
        assert len(np.unique(np.r_[ix['context'],ix['correction'],ix['validation']]))==70000
        supports.append({'fold':fold,'outer_positive':int(y[ix['validation']].sum()),'outer_negative':int((1-y[ix['validation']]).sum())})
    for pack in plan['pilot_packs']:
        pred=np.full(len(y),np.nan)
        for fold in range(5):
            root=R/f'runs/small2/{pack}/fold{fold}';done=json.loads((root/'RESULT.json').read_text())
            assert done['status']=='complete' and done['replay_max_abs']<1e-7
            assert sha(root/'prediction.npz')==done['prediction_sha256']
            z=np.load(root/'prediction.npz');rows=z['rows'][6000:]
            assert np.array_equal(rows,np.flatnonzero(folds==fold))
            assert np.isfinite(z['raw']).all() and len(z['raw'])==20000
            pred[rows]=z['raw'][6000:];seconds.append(done['seconds'])
        metrics=score(y,pred)
        assert abs(metrics['KS']-ks_2samp(pred[y==0],pred[y==1]).statistic)<1e-12
        metrics['fold_KS']=[score(y[folds==fold],pred[folds==fold])['KS'] for fold in range(5)]
        results[pack]=metrics
    # Timing evidence cannot establish upstream capture semantics; preserve this limit.
    timing=json.loads((R/'REPAYMENT_TIMING_AUDIT.json').read_text())
    assert all(timing[k]==0 for k in timing if k!='rows')
    data_gate=json.loads((R/'DATA_GATE.json').read_text())
    assert data_gate['outer_labels_copied'] is False
    all_ids={s['id'] for s in catalog['feature_definitions']}
    result={'status':'passed_for_declared_development_experiment','catalog_sha256':plan['catalog_sha256'],
        'formula_checks':unit_checks,'evaluator_known_cases':True,'recomputed_KS_matches_two_sample_KS':True,
        'supports':supports,'pilot_metrics':results,'median_fit_seconds':float(np.median(seconds)),
        'features_after_fold0_constant_duplicate_filter':len(covered),'features_excluded_fold0':len(all_ids-covered),
        'excluded_feature_ids':sorted(all_ids-covered),'pack_widths':finite,'drops':dropped,
        'retention':'No fitted weights or KV caches','independent_confirmation':False,
        'limits':['All outer labels have prior development exposure.',
                  'Timestamp and DPD consistency checked; upstream repayment snapshot semantics remain unverified.',
                  'Catalog formulas are hypotheses; sources do not establish gains in this competition.'],
        'provenance':{'source_sha256':sha(__file__),'plan_sha256':sha(R/'PLAN.json'),'data_gate_sha256':sha(R/'DATA_GATE.json'),
                      'timing_audit_sha256':sha(R/'REPAYMENT_TIMING_AUDIT.json')},'seconds':time.time()-started}
    (R/'PILOT_GATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['status','pilot_metrics','median_fit_seconds','features_after_fold0_constant_duplicate_filter','features_excluded_fold0']},indent=2))
if __name__=='__main__':main()
