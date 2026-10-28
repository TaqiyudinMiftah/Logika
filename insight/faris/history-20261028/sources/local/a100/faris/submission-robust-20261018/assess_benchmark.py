"""Report uncertainty for the one frozen nested selector; does not retune it."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score,log_loss

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'submission-final-20261018'))
from audit_generalization import paired_bootstrap


def mean_fold_interval(y,a,b,folds,repetitions=2000):
    from audit_generalization import ordered_statistic,weighted_ks
    rng=np.random.default_rng(20261020);draws=np.zeros(repetitions)
    for f in range(3):
        mask=folds==f;t=y[mask];aa=a[mask];bb=b[mask]
        order_a=ordered_statistic(t,aa);order_b=ordered_statistic(t,bb)
        classes=[np.flatnonzero(t==c) for c in [0,1]]
        for i in range(repetitions):
            ids=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in classes]);w=np.bincount(ids,minlength=len(t))
            draws[i]+=(weighted_ks(t,w,order_a)-weighted_ks(t,w,order_b))/3
    return {'paired_stratified_95ci':np.quantile(draws,[.025,.975]).tolist(),'samples':repetitions,
        'metric':'Mean of the three within-outer-fold KS changes',
        'caveat':'Conditional on the frozen historical shortlist and fitted fold models; does not erase historical reuse or measure full-refit variance.'}


def main():
    z=np.load(ROOT/'evidence/nested_oof.npz');y=z['target'];a=z['prediction'];b=z['champion']
    uncertainty=paired_bootstrap(y,a,b,repetitions=2000,seed=20261020)
    plan=json.loads((ROOT/'benchmark_plan.json').read_text());train=pd.read_csv(plan['train']);months=pd.to_datetime(train.decision_datetime.iloc[z['rows']]).dt.strftime('%Y-%m').to_numpy();cohorts=[]
    for month in sorted(set(months)):
        mask=months==month
        score=lambda p:float(ks_2samp(p[mask&(y==0)],p[mask&(y==1)]).statistic)
        cohorts.append({'month':month,'n':int(mask.sum()),'delta':score(a)-score(b)})
    report={'primary':'One frozen nested addition-and-weight selection procedure versus champion',
        'primary_mean_fold_uncertainty':mean_fold_interval(y,a,b,z['fold']),
        'uncertainty':uncertainty,'auc_delta':float(roc_auc_score(y,a)-roc_auc_score(y,b)),
        'log_loss_delta':float(log_loss(y,np.clip(a,1e-8,1-1e-8))-log_loss(y,np.clip(b,1e-8,1-1e-8))),
        'monthly_diagnostics':cohorts,'monthly_diagnostics_do_not_change_selection':True,
        'historical_labels_used_for_discovery':True,'untouched_private_estimate':False,
        'new_outer_prediction_coverage':len(y),'all_rows_finite':bool(np.isfinite(a).all()),
        'no_confirmation_rescoring':True,'no_upload':True}
    (ROOT/'benchmark_assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
