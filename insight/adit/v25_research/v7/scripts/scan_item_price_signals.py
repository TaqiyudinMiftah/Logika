"""Descriptive H1/H2 data-pattern checks, not independent validation."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'


def main():
    b=pd.read_csv(V/'runs/V25/OOF.csv',float_precision='round_trip')
    f=pd.read_parquet(V/'data_fe/data/item_price_resilience.parquet').set_index('application_id').reindex(b.application_id)
    meta=pd.read_parquet(V/'audit_noise/data/application_metadata.parquet',columns=['application_id','decision_datetime']).set_index('application_id').reindex(b.application_id)
    y=b.target.to_numpy(int);p=b.prediction.to_numpy();h1=pd.to_datetime(meta.decision_datetime).dt.month.to_numpy()<=6
    rows=[];summary=[]
    for c in f:
        a=f[c].to_numpy(float);finite=np.isfinite(a);cuts=np.unique(np.quantile(a[finite&h1],[.25,.5,.75]));groups=np.where(finite,np.searchsorted(cuts,a,side='right'),-1)
        for label,half in [('H1',h1),('H2',~h1)]:
            ix=finite&half
            summary.append(dict(feature=c,half=label,finite=int(ix.sum()),AUC=float(roc_auc_score(y[ix],a[ix])) if len(np.unique(a[ix]))>1 else .5,KS=float(ks_2samp(a[ix&(y==0)],a[ix&(y==1)]).statistic)))
            for group in sorted(set(groups)):
                ix=(groups==group)&half
                if not ix.any():continue
                residual=y[ix]-p[ix];se=float(residual.std(ddof=1)/np.sqrt(ix.sum()))
                rows.append(dict(feature=c,half=label,bin=int(group),cuts=cuts.tolist(),n=int(ix.sum()),events=int(y[ix].sum()),default_rate=float(y[ix].mean()),mean_V25=float(p[ix].mean()),excess_default=float(residual.mean()),excess_CI_low=float(residual.mean()-1.96*se),excess_CI_high=float(residual.mean()+1.96*se)))
    out=V/'data_fe/reports';pd.DataFrame(rows).to_csv(out/'HIDDEN_SIGNAL_BINS.csv',index=False);pd.DataFrame(summary).to_csv(out/'HIDDEN_SIGNAL_UNIVARIATE.csv',index=False)
    print('DESCRIPTIVE_SIGNAL_SCAN_COMPLETE',len(summary),'rows; intervals exclude repeated model/feature selection and are not evidence of unique predictive value',flush=True)


if __name__=='__main__':main()
