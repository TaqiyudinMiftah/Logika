"""Matched domain-specific smoothing after V200's AUC/KS tradeoff."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse,io,json,sys,traceback
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import dump,sha,design,corrected_probability
from experiments import fit_head
from conservative_adapters import main as evaluate
from stability_worker import solve_quadratic_newton,postprocess
SCOPES=dict(capacity=['re_income_geometric_mean','re_declared_income_log_gap','re_outflow_income_geometric_ratio','installment_observed_income_ratio','v2_existing_total_burden_observed'],evidence=['re_price_posterior_sd','re_bureau_posterior_sd','risk_conditional_negative_months','risk_conditional_nonnegative_months','aux_account_months','aux_orders90','aux_ecommerce_observed_days','aux_allocation_count'])
SCOPES['credit_signal']=['re_bureau_score_posterior','re_late_1_posterior','re_late_30_posterior']


def fit_adapter(f,spec,columns):
    head=fit_head(f['h'],f['kh'],columns,.003,.5);x,z=design(head,f['h'],f['kh']);numeric=list(head['encoder'].transformers_[0][2]);smooth=head['encoder'].named_transformers_['smooth'];width=smooth.named_steps['basis'].n_features_out_//len(numeric);scale=smooth.named_steps['scale'].scale_;selected=SCOPES[spec['roughness_scope']];assert set(selected)<=set(numeric);rows=[]
    for j,name in enumerate(numeric):
        if name not in selected:continue
        for q in range(width-2):
            row=np.zeros(x.shape[1]);pos=j*width+q;row[2+pos:2+pos+3]=np.array([1.,-2.,1.])/scale[pos:pos+3];rows.append(row)
    difference=np.array(rows);penalty=.003*np.eye(x.shape[1]);penalty[0,0]=0.;penalty+=spec['roughness_penalty']*(difference.T@difference)
    head['coefficients'],audit=solve_quadratic_newton(head['coefficients'],x,z,f['h'].target.to_numpy(int),penalty);audit.update(roughness_scope=spec['roughness_scope'],smoothed_inputs=selected,roughness_rows=len(rows))
    return dict(head=head,penalty=penalty,scope=selected),audit,len(head['coefficients'])


def probability(model,frame,k,base_head):return corrected_probability(model['head'],frame,k)


def preflight():
    rng=np.random.default_rng(56027);cols=json.loads((V/'runs/V25/RESULT.json').read_text())['columns'];n=1000;h=pd.DataFrame({c:rng.lognormal(size=n) for c in cols if c!='employment_type'});h['employment_type']=np.resize(['salaried','self_employed','variable_income'],n);k=rng.beta(2,8,n);h['target']=rng.binomial(1,k);checks=[]
    for spec in json.loads((V/'scoped_smoothing_plan.json').read_text()):
        f=dict(h=h,kh=k);model,audit,count=fit_adapter(f,spec,cols);p,_=probability(model,h.iloc[:40],k[:40],None);assert np.isfinite(p).all();assert audit['roughness_rows']==2*len(SCOPES[spec['roughness_scope']])
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0);p2,_=probability(joblib.load(stream),h.iloc[:40],k[:40],None);assert np.array_equal(p,p2);checks.append(dict(scope=spec['roughness_scope'],rows=audit['roughness_rows'],replay_exact=True))
    dump(dict(status='PASS',synthetic_only=True,checks=checks,source_sha256=sha(__file__)),V/'scoped_smoothing/reports/PREFLIGHT.json');print('SCOPED_PREFLIGHT_PASS',flush=True)


def main(ident):
    spec=next(s for s in json.loads((V/'scoped_smoothing_plan.json').read_text()) if s['id']==ident);pre=json.loads((V/'scoped_smoothing/reports/PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['source_sha256']==sha(__file__)
    return evaluate(ident,plan_file='scoped_smoothing_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,method_formulas={spec['method']:spec['formula']},spectral_methods=[],worker_source='scoped_smoothing.py',postprocess=postprocess)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    try:
        if args.preflight:preflight()
        else:assert args.run;main(args.run)
    except Exception:
        if args.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json')
        raise
