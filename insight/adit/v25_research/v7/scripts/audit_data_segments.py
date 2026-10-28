"""Domain and source-coverage audit of V221 vs V25, without new fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
ROOT = Path(__file__).resolve().parents[2]
V = ROOT/'v7'
OUT = V/'audit_noise/reports'


def metrics(y, p):
    f,t,_=roc_curve(y,p,drop_intermediate=False)
    return dict(KS=float(np.max(abs(t-f))),AUC=float(roc_auc_score(y,p)))


def main():
    tr=pd.read_parquet(ROOT/'data/train_features.parquet')
    te=pd.read_parquet(ROOT/'data/test_features.parquet')
    b=pd.read_csv(V/'runs/V25/OOF.csv',float_precision='round_trip')
    p=pd.read_csv(V/'runs/V221/OOF.csv',float_precision='round_trip')
    assert tr.application_id.tolist()==b.application_id.tolist()==p.application_id.tolist()
    y=tr.target.to_numpy(int);base=b.prediction.to_numpy();pred=p.prediction.to_numpy()
    support=['aux_orders90','aux_allocation_count','aux_account_months','risk_conditional_negative_months','risk_conditional_nonnegative_months','aux_ecommerce_observed_days']
    parts=[]
    for fold in range(5):
        parts.append(pd.read_parquet(V/'extension/data'/f'fold{fold}/validation.parquet',columns=['application_id']+support))
    extra=pd.concat(parts).set_index('application_id')
    assert extra.index.is_unique and len(extra)==len(tr)
    extra=extra.reindex(tr.application_id)
    test_extra=pd.read_parquet(V/'extension/data/fold0/test.parquet',columns=['application_id']+support).set_index('application_id').reindex(te.application_id)
    spectral=pd.read_parquet(V/'light_fe/data/spectral_features.parquet').set_index('application_id').reindex(tr.application_id)
    dx=(rankdata(pred)-rankdata(base))/len(y)
    order_band=lambda d:np.select([d.aux_orders90<=0,d.aux_orders90<=3,d.aux_orders90<=10],['zero','1to3','4to10'],default='over10')
    g,gt=order_band(extra),order_band(test_extra)
    account=extra.aux_account_months.to_numpy()>0
    account_t=test_extra.aux_account_months.to_numpy()>0
    allocated=extra.aux_allocation_count.to_numpy()>0
    allocated_t=test_extra.aux_allocation_count.to_numpy()>0
    masks=[]
    for name,group,group_t in [('account',account,account_t),('allocation',allocated,allocated_t)]:
        for flag in [False,True]:
            masks.append((name,'observed' if flag else 'none',group==flag,group_t==flag))
            for band in ['zero','1to3','4to10','over10']:
                masks.append((name+'_x_orders90',('observed' if flag else 'none')+'_'+band,(group==flag)&(g==band),(group_t==flag)&(gt==band)))
    for band in ['zero','1to3','4to10','over10']:
        masks.append(('orders90',band,g==band,gt==band))
    rows=[]
    for name,label,ix,it in masks:
        if ix.sum()<50 or len(np.unique(y[ix]))<2:continue
        a,c=metrics(y[ix],base[ix]),metrics(y[ix],pred[ix])
        rows.append(dict(segment=name,value=label,n=int(ix.sum()),events=int(y[ix].sum()),target_rate=float(y[ix].mean()),
                         train_share=float(ix.mean()),test_share=float(it.mean()),test_n=int(it.sum()),
                         KS_V25=a['KS'],KS_V221=c['KS'],delta_KS=c['KS']-a['KS'],delta_AUC=c['AUC']-a['AUC'],
                         mean_rank_change=float(dx[ix].mean()),
                         spectral_spend_strength_median=float(spectral.loc[ix,'beh_spectral_spend_periodicity_strength'].median()),
                         spectral_finite_share=float(spectral.loc[ix,'beh_spectral_spend_periodicity_strength'].notna().mean())))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'COVERAGE_ACTIVITY_SEGMENTS.csv',index=False)
    correlations=[]
    for col in spectral:
        a=spectral[col].to_numpy(float);count=extra.aux_orders90.to_numpy(float);ok=np.isfinite(a)&np.isfinite(count)
        if np.unique(a[ok]).size<2:continue
        correlations.append(dict(feature=col,spearman_orders90=float(spearmanr(a[ok],count[ok]).statistic),
                                 spearman_target=float(spearmanr(a[ok],y[ok]).statistic),
                                 spearman_V221_logit_change=float(spearmanr(a[ok],(logit(np.clip(pred,1e-6,1-1e-6))-logit(np.clip(base,1e-6,1-1e-6)))[ok]).statistic)))
    pd.DataFrame(correlations).to_csv(OUT/'SPECTRAL_DEPENDENCE.csv',index=False)
    lineage=[]
    for ident in ['V25','V167','V178','V189','V193','V219','V221']:
        r=json.loads((V/'runs'/ident/'RESULT.json').read_text())
        lineage.append(dict(model=ident,method=r.get('modeling_formula'),columns=r['columns'],
                            logged_features_changed_vs_V25=r.get('features_changed_vs_V25'),
                            logged_parameters=[f.get('parameters') for f in r['folds']],
                            logged_new_trainable_parameters=[f.get('new_trainable_parameters') for f in r['folds']],
                            head_gap=float(np.mean([f['train_validation_head_KS_gap'] for f in r['folds']])),
                            caveat=r['folds'][0].get('head_gap_caveat')))
    value=dict(segments=rows,spectral_dependencies=correlations,lineage=lineage,
               source_warning='V221 metadata only lists its immediate fixed-fusion inputs and inherits the V25 head gap. Its prediction lineage includes V167 spectral145-parameter correction and V219 three bootstrap heads; no newly fitted fusion parameters does not mean low-complexity total ensemble.',
               causation_warning='Exploratory source/coverage associations; no hidden test labels. Segment differences do not identify the cause of public-score changes. Spectral strength is normalized periodogram power, not proven periodic financial behavior.',
               code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'DATA_SEGMENT_AUDIT.json').write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    print(frame[frame.segment.isin(['account','allocation','orders90'])].to_string(index=False))
    print('SPECTRAL_DEPENDENCE',json.dumps(correlations))


if __name__=='__main__':main()
