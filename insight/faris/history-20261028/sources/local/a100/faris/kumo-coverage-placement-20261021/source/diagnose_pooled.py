"""Descriptive ROC diagnosis of completed OOF evidence; no fitted candidate."""
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
from scipy.stats import ks_2samp
R=Path(__file__).resolve().parents[1]
z=np.load(R/'oof.npz');y=z['target'];fold=z['fold'];out={}
for variant in ['control90','coverage96','null96']:
 p=z[variant];fpr,tpr,threshold=roc_curve(y,p,drop_intermediate=False);d=tpr-fpr;i=np.argmax(d)
 assert abs(d[i]-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 # Evaluate at the reference's single global threshold, never optimize routing.
 ref=z['control90'];rf,rt,rth=roc_curve(y,ref,drop_intermediate=False);cut=rth[np.argmax(rt-rf)]
 rows=[]
 for f in range(5):
  take=fold==f;yy=y[take];pp=p[take];a,b,c=roc_curve(yy,pp,drop_intermediate=False);j=np.argmax(b-a)
  rows.append(dict(fold=f,KS=float((b-a)[j]),own_best_threshold=float(c[j]),
    KS_at_own_global_threshold=float(np.mean(pp[yy==1]>=threshold[i])-np.mean(pp[yy==0]>=threshold[i])),
    KS_at_reference_global_threshold=float(np.mean(pp[yy==1]>=cut)-np.mean(pp[yy==0]>=cut)),
    mean_probability=float(pp.mean()),prevalence=float(yy.mean()),quantiles=np.quantile(pp,[.1,.5,.9]).tolist()))
 pair=np.zeros((5,5));weights=np.zeros((5,5))
 for a in range(5):
  pos=p[(y==1)&(fold==a)]
  for b in range(5):
   neg=np.sort(p[(y==0)&(fold==b)]);l=np.searchsorted(neg,pos,side='left');r=np.searchsorted(neg,pos,side='right')
   pair[a,b]=np.mean((l+r)/2)/len(neg);weights[a,b]=len(pos)*len(neg)
 assert abs(np.average(pair,weights=weights)-roc_auc_score(y,p))<1e-12
 grid=np.linspace(0,1,10001);curve=np.interp(grid,fpr,tpr)
 out[variant]=dict(pooled_KS=float(d[i]),AUC=float(roc_auc_score(y,p)),best_threshold=float(threshold[i]),
  FPR_at_KS=float(fpr[i]),TPR_at_KS=float(tpr[i]),folds=rows,positive_fold_negative_fold_AUC=pair.tolist(),
  within_fold_AUC=float(np.average(np.diag(pair),weights=np.diag(weights))),
  cross_fold_AUC=float(np.average(pair[~np.eye(5,dtype=bool)],weights=weights[~np.eye(5,dtype=bool)])),
  mean_fold_KS=float(np.mean([r['KS']for r in rows])),
  mean_TPR_FPR_05_30=float(np.mean(curve[(grid>=.05)&(grid<=.30)]-grid[(grid>=.05)&(grid<=.30)])))
report=dict(status='descriptive_complete',recipes=out,OOF_sha256=hashlib.sha256((R/'oof.npz').read_bytes()).hexdigest(),
 no_new_training=True,no_new_recipe=True,no_label_derived_threshold_or_routing_deployed=True,
 caveat='Fold maxima and pooled maximum occur at different thresholds. Descriptive inspection cannot establish fold calibration as causal or estimate fresh generalization.')
(R/'POOLED_DIAGNOSIS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{a:v[a]for a in ['pooled_KS','AUC','best_threshold','within_fold_AUC','cross_fold_AUC','mean_fold_KS','mean_TPR_FPR_05_30']}for k,v in out.items()},indent=2))
