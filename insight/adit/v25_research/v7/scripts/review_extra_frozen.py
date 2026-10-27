"""Data/reference sanity review, not a new model-performance experiment."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import scores,sha
from extra_frozen_context import sources,prepare,fit_fold

def main():
    try:Path('/proc/self/oom_score_adj').write_text('750')
    except OSError:pass
    out=V/'kumo_gpu/extra_review';out.mkdir(parents=True,exist_ok=True)
    definition=json.loads((ROOT/'provenance/definition.json').read_text());latent=[c for c in definition['columns'] if c.startswith('re_')]
    core=['installment_observed_income_ratio','v2_ecom_30d_orders','v3_ecom_category_entropy','v2_ecom_30d_spend','v2_bureau_history_score_mean','bureau_score_mean','v3_repay_all_late_1_posterior_24','v4_bureau_all_equal_provider_mean_std','v5_price_pooled_all_mean']
    tr=pd.read_parquet(ROOT/'data/train_features.parquet',columns=['application_id','target']+core+latent)
    te=pd.read_parquet(ROOT/'data/test_features.parquet',columns=['application_id']+core+latent)
    specs=[s for s in json.loads((V/'kumo_plan.json').read_text()) if int(s['id'][1:])>=174]
    extra=list(dict.fromkeys(['aux_allocation_count','aux_Bureau_A','aux_Bureau_B','aux_Bureau_C']+[c for s in specs for c in sources(s)]))
    raw=pd.read_parquet(V/'data/features_raw.parquet',columns=['application_id']+extra)
    tr=tr.merge(raw,on='application_id',validate='one_to_one');te=te.merge(raw,on='application_id',validate='one_to_one')
    idx=np.load(ROOT/'artifacts/fold0/indices.npz');ci,vi=idx['context'],idx['validation']
    checks=[]
    for s in specs:
        names,formulas=prepare(s,tr,te,V);a=fit_fold(s,tr,te,ci,out,0);observations=[]
        for c in names:
            x=tr[c].to_numpy(float);ok=np.isfinite(x[vi]);row=dict(feature=c,observed_validation=int(ok.sum()),missing_test=int(te[c].isna().sum()))
            if min(int(tr.target.to_numpy()[vi][ok].sum()),int(ok.sum()-tr.target.to_numpy()[vi][ok].sum()))>=25:
                row['univariate_heldout']=scores(tr.target.to_numpy()[vi][ok],x[vi][ok])
            correlations=[]
            for existing in core:
                xx=tr[existing].to_numpy(float);keep=np.isfinite(x[ci])&np.isfinite(xx[ci])
                if keep.sum()>100 and np.std(x[ci][keep])>1e-8 and np.std(xx[ci][keep])>1e-8:
                    value=float(spearmanr(x[ci][keep],xx[ci][keep]).statistic)
                    if np.isfinite(value):correlations.append((existing,value))
            row['strongest_existing_correlation']=sorted(correlations,key=lambda q:abs(q[1]),reverse=True)[:2];observations.append(row)
        checks.append(dict(id=s['id'],hypothesis=s['error_analysis_basis'],input_count=len(names),formula_checks='PASS',reference=a,descriptive_validation_statistics=observations))
    provider=tr[['aux_Bureau_A','aux_Bureau_B','aux_Bureau_C']].iloc[ci]
    value=dict(status='PASS',scope='Feature/availability/reference/finite-division checks, plus exploratory fold0 univariate signal and redundancy. No Kumo fitting, no claimed incremental KS.',reference_rows=len(ci),heldout_rows=len(vi),checks=checks,provider_context_mean=provider.mean().to_dict(),provider_context_sd=provider.std().to_dict(),snapshot_provenance='Inherited baseline uncertainties remain; allocation is last-allocation timing, not DPD',source_sha256=sha(__file__))
    (out/'REVIEW.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)

if __name__=='__main__':main()
