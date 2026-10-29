"""Replay saved V262 heads, inner teachers, fold scores and delivered CSV."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import json,sys,hashlib,shutil,csv
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(V/'scripts'),str(ROOT/'scripts')]
from modeling_features import load
from credit_posthoc_worker import probability,TERMS
from baseline import dump,sha,corrected_probability
from audit_frozen_result import audit


def main():
    out=V/'runs/V262';result=json.loads((out/'RESULT.json').read_text())
    assert sha(V/'scripts/credit_posthoc_worker.py')==result['additional_source_sha256']
    tr,te,sample,y,folds,base,frames,parent=load();checks=[]
    for f in frames:
        saved=joblib.load(out/f"fold{f['fold']}_correction.joblib");model=saved['adapter']
        assert np.array_equal(saved['base_head']['coefficients'],f['head']['coefficients'])
        inner=[]
        for block,head in enumerate(model['inner_heads']):
            qi=np.flatnonzero(model['inner_assignments']==block)
            ti=np.flatnonzero(model['inner_assignments']!=block)
            assert (len(ti),len(qi))==(4000,2000)
            rec=result['folds'][f['fold']]['optimization']['inner_blocks'][block]
            for label,idx in [('fit',ti),('query',qi)]:
                digest=hashlib.sha256('\n'.join(f['h'].application_id.iloc[idx].astype(str)).encode()).hexdigest()
                assert digest==rec[label+'_ids_sha256']
            p,_=corrected_probability(head,f['h'].iloc[qi],f['kh'][qi])
            error=float(np.max(abs(p-model['inner_teacher_predictions'][qi])))
            assert error<1e-14;inner.append(error)
        errors={};maxdelta=0.
        for part,cache,spline in [('val','kv','sv'),('test','kt','st')]:
            p,d=probability(model,f[part],f[cache],f['head'])
            prediction=.75*p+.25*f[spline]
            path=out/f"fold{f['fold']}_{'validation' if part=='val' else 'test'}.parquet"
            old=pd.read_parquet(path)
            assert old.application_id.tolist()==f[part].application_id.tolist()
            error=float(np.max(abs(prediction-old.prediction.to_numpy())))
            assert error<1e-14;errors[part]=error
            maxdelta=max(maxdelta,float(np.max(abs(d))))
        checks.append(dict(fold=f['fold'],inner_teacher_replay_max_abs=inner,
                           prediction_replay_max_abs=errors,max_added_logit=maxdelta,
                           base_head_coefficients_unchanged=True))
    measured=audit('V262')
    submission=ROOT/result['submission']
    delivered=Path.home()/'Downloads/submission_V262_V25_credit_posthoc.csv'
    if delivered.exists():assert sha(delivered)==sha(submission),'Do not overwrite an unrelated existing file'
    else:shutil.copyfile(submission,delivered)
    assert sha(delivered)==sha(submission)
    receipt=dict(status='PASS',experiment='V262',folds=checks,
                 all_checkpoint_predictions_replayed=True,inner_teacher_isolation_hashes_match=True,
                 original_heads_unchanged=True,source_sha256=sha(__file__),
                 submission_sha256=sha(submission),delivered_file=str(delivered),
                 rows=30000,no_kaggle_upload=True,paired_delta_CI=result['confidence_interval']['delta_KS'])
    dump(receipt,out/'POSTHOC_REPLAY_AND_DELIVERY.json')
    formulas=[
        'z(late30)*(n_allocation/(n_allocation+10)-train_mean_reliability); n_allocation>0',
        'z(installment/income)*(q-mu)*m/(m+3); q=(negative+2mu)/(m+2); m=negative+nonnegative>0',
        'z(installment/income)*z(outflow/income)*account_months/(account_months+3); observed support only',
        'z(total_burden/income)*(train_mean_reliability-n_allocation/(n_allocation+10)); observed count including0']
    catalog=V/'credit_posthoc/reports/FEATURE_CATALOG.csv'
    with catalog.open('w',newline='',encoding='utf-8') as stream:
        fields=['feature_name','family','source_tables','formula','cutoff_rule','validation_status','evidence','leakage_risk','priority']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,formula in zip(TERMS,formulas):
            writer.writerow(dict(feature_name=name,family='credit_posthoc',source_tables='Existing V25 application/repayment/account feature cache',formula=formula+'; train6k signedlog reference; support mask; main-effect ridge residualization .001; clip[-4,4]',cutoff_rule='Inherits V25 historical cutoffs and unresolved snapshot provenance; references fit correction-training only',validation_status='NO_ESTABLISHED_GROUP_GAIN',evidence=f"Joint V262 deltaKS={result['delta_KS']};95%CI={result['confidence_interval']['delta_KS']}; individual utility not isolated",leakage_risk='Inherited legacy DPD uncertainty; no new rows/timestamps; honest inner teacher; repeated-development selection risk',priority='P2'))
    print(json.dumps(dict(status='PASS',rows=30000,delivered=str(delivered),KS=result['OOF']['KS'],AUC=result['OOF']['AUC'],sha256=sha(delivered))))


if __name__=='__main__':main()
