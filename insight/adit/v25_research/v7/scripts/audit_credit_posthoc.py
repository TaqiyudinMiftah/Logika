"""Read-only diagnostic of existing held-out scores and conditional credit errors.

No new model, score correction, feature selection or submission is fitted here.
All bins/contrasts are exploratory on historically reused development data.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7';sys.path.insert(0,str(ROOT/'scripts'))
from baseline import scores,sha,dump


def wilson(y):
    n=len(y);p=float(np.mean(y));z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]


def summary(d):
    residual=d.target.to_numpy()-d.prediction.to_numpy();n=len(d);se=float(np.std(residual,ddof=1)/np.sqrt(n)) if n>1 else 0.
    return dict(n=n,events=int(d.target.sum()),default_rate=float(d.target.mean()),default_rate_Wilson95=wilson(d.target),mean_prediction=float(d.prediction.mean()),mean_residual=float(residual.mean()),residual95=[float(residual.mean()-1.96*se),float(residual.mean()+1.96*se)])


def main():
    out=V/'credit_posthoc/reports';out.mkdir(parents=True,exist_ok=True)
    tr=pd.read_parquet(ROOT/'data/train_features.parquet',columns=['application_id','customer_id','target']);te=pd.read_parquet(ROOT/'data/test_features.parquet',columns=['application_id','customer_id'])
    oof=pd.read_csv(V/'runs/V25/OOF.csv',float_precision='round_trip');assert oof.application_id.tolist()==tr.application_id.tolist() and np.array_equal(oof.target,tr.target)
    inputs=V/'anchored_spline/data/inputs.parquet';features=pd.read_parquet(inputs);d=oof[['application_id','target','fold','prediction']].merge(features,on='application_id',validate='one_to_one');assert len(d)==len(tr)
    raw=np.full(len(tr),np.nan);spline=raw.copy();fold_checks=[]
    for fold in range(5):
        idx=np.load(ROOT/'artifacts'/f'fold{fold}'/'indices.npz');ci,hi,vi=[idx[k] for k in ['context','correction','validation']]
        assert len(set(np.r_[ci,hi,vi]))==len(tr)
        k=pd.read_parquet(ROOT/'artifacts'/f'fold{fold}'/'kumo_oof.parquet');s=pd.read_parquet(ROOT/'artifacts'/f'fold{fold}'/'spline_oof.parquet')
        assert k.application_id.tolist()==s.application_id.tolist()==tr.application_id.iloc[vi].tolist()
        raw[vi]=k.prediction;spline[vi]=s.prediction
        fold_checks.append(dict(fold=fold,context=len(ci),correction=len(hi),validation=len(vi)))
    pure=scores(tr.target,raw);before=scores(tr.target,.75*raw+.25*spline);v25=scores(tr.target,d.prediction)
    old=pd.read_csv(ROOT/'reports/OOF.csv',float_precision='round_trip');assert old.application_id.tolist()==tr.application_id.tolist()
    original=scores(tr.target,old.prediction)
    monotone=expit(.7*logit(d.prediction.to_numpy())+.2);mapped=scores(tr.target,monotone)
    assert max(abs(mapped[k]-v25[k]) for k in ['KS','AUC'])<1e-14
    conditions={
        'bureau_posterior':d.re_bureau_score_posterior,
        'late30_posterior':d.re_late_30_posterior,
        'installment_observed_income':d.installment_observed_income_ratio,
        'total_burden_observed':d.v2_existing_total_burden_observed,
        'price_posterior':d.re_price_log_posterior,
        'bureau_uncertainty':d.re_bureau_posterior_sd,
        'declared_income_gap':d.re_declared_income_log_gap}
    masks={};cutoffs={}
    for key,a in conditions.items():
        lo,hi=np.nanquantile(a,[.25,.75]);cutoffs[key]=[float(lo),float(hi)]
        masks[key+'_low']=a.le(lo);masks[key+'_high']=a.gt(hi)
    masks['no_account']=d.aux_account_months.eq(0);masks['has_account']=d.aux_account_months.gt(0)
    masks['no_allocation']=d.aux_allocation_count.eq(0);masks['allocation_1to3']=d.aux_allocation_count.between(1,3);masks['allocation_10plus']=d.aux_allocation_count.ge(10)
    masks['healthy_bureau_bad_repayment']=masks['bureau_posterior_high']&masks['late30_posterior_high']
    masks['weak_bureau_good_repayment']=masks['bureau_posterior_low']&masks['late30_posterior_low']
    masks['high_burden_sparse_history']=masks['total_burden_observed_high']&d.aux_allocation_count.le(3)
    masks['high_burden_rich_history']=masks['total_burden_observed_high']&d.aux_allocation_count.ge(10)
    d['risk_band']=pd.cut(d.prediction,[0,.05,.1,.2,.4,1],include_lowest=True).astype(str)
    segments=[]
    for name,mask in masks.items():
        for band in ['ALL']+sorted(d.risk_band.unique()):
            ix=mask if band=='ALL' else mask&d.risk_band.eq(band)
            if ix.sum()<200:continue
            seg=d.loc[ix];s=summary(seg);s.update(segment=name,risk_band=band,fold_residual_means=[summary(seg[seg.fold==f])['mean_residual'] if (seg.fold==f).sum()>30 else None for f in range(5)])
            segments.append(s)
    # Equalize the comparison by fold and fine score band. This is a diagnostic,
    # not causal adjustment or a fitted reranking rule.
    d['score_decile']=pd.qcut(d.prediction,20,labels=False,duplicates='drop')
    contrasts=[]
    for name in conditions:
        a=masks[name+'_high'];b=masks[name+'_low'];cells=[]
        for fold in range(5):
            for band in sorted(d.score_decile.unique()):
                base=d.fold.eq(fold)&d.score_decile.eq(band);left=d[base&a];right=d[base&b]
                if min(len(left),len(right))<20:continue
                u=left.target-left.prediction;w=right.target-right.prediction
                cells.append(dict(fold=fold,n_high=len(left),n_low=len(right),weight=min(len(left),len(right)),delta=float(u.mean()-w.mean()),variance=float(u.var(ddof=1)/len(u)+w.var(ddof=1)/len(w))))
        if not cells:continue
        weights=np.array([c['weight'] for c in cells],float);weights/=weights.sum();delta=float(weights@np.array([c['delta'] for c in cells]));se=float(np.sqrt(weights**2@np.array([c['variance'] for c in cells])))
        byfold=[]
        for fold in range(5):
            s=[c for c in cells if c['fold']==fold];byfold.append(float(np.average([c['delta'] for c in s],weights=[c['weight'] for c in s])) if s else None)
        contrasts.append(dict(variable=name,high_minus_low_residual=delta,diagnostic95=[delta-1.96*se,delta+1.96*se],raw_two_sided_p=float(2*norm.sf(abs(delta)/se)) if se>0 else None,byfold=byfold,cells=len(cells),compared_high_rows=sum(c['n_high'] for c in cells),compared_low_rows=sum(c['n_low'] for c in cells)))
    m=len(contrasts)
    for row in contrasts:row['bonferroni_p_seven_contrasts']=min(1,row['raw_two_sided_p']*m) if row['raw_two_sided_p'] is not None else None
    result=dict(status='COMPLETE_READ_ONLY',rows=len(tr),events=int(tr.target.sum()),target_rate=float(tr.target.mean()),test_rows=len(te),unique_customer_train=tr.customer_id.nunique(),customer_overlap=len(set(tr.customer_id)&set(te.customer_id)),folds=fold_checks,
        metrics=dict(frozen_Kumo_raw=pure,raw_Kumo75_spline25=before,original12_input_offset=original,V25_18_input_offset=v25,global_monotone_remapping=mapped),
        source_sha256=sha(__file__),input_sha256=sha(inputs),OOF_sha256=sha(V/'runs/V25/OOF.csv'),bins=cutoffs,contrasts=contrasts,
        caveats=['Exploratory historically reused development OOF, not independent confirmation','Score-band conditioning is descriptive, not causal; model errors remain dependent through training and unmeasured confounding','Intervals are approximate sampling uncertainty conditional on fitted scores; no retraining/search adjustment','Bonferroni applies only to these seven descriptive contrasts, not the whole research history','Source snapshot/global-reference uncertainty inherited; no new model, posthoc correction, or submission produced'])
    pd.DataFrame(segments).to_csv(out/'CREDIT_RESIDUAL_SEGMENTS.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'MATCHED_SCORE_CONTRASTS.csv',index=False);dump(result,out/'CREDIT_POSTHOC_AUDIT.json')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
