"""Analytic conditional correction variance; no evaluation labels are opened."""
from pathlib import Path
import sys,json,numpy as np,pandas as pd,joblib
from scipy.special import expit
R=Path(__file__).resolve().parent;A=R.parent
sys.path.insert(0,str(A/'kumo-coverage-placement-20261021/source'))
import baseline as b
out=[]
for f in range(5):
 d=A/f'kumo-coverage-placement-20261021/conditions/fold{f}/control90';c=json.loads((d/'complete.json').read_text())
 for n in ['correction.joblib','raw_predictions.npz']:assert b.sha(d/n)==c['small_files'][n]
 model=joblib.load(d/'correction.joblib');raw=np.load(d/'raw_predictions.npz')['prediction'];data=A/f'kumo-temperature-adaptation-20261021/full_data/fold{f}';q=pd.read_parquet(data/'query.parquet');y=np.load(data/'correction_labels.npy').astype(np.int64);assert len(y)==6000 and'target'not in q
 x,o=b.design(model,q.iloc[:6000],raw[:6000]);xv,ov=b.design(model,q.iloc[6000:],raw[6000:]);coef=model['coefficients'];eta=o+x@coef;pr=expit(eta);n=len(y);pen=np.eye(x.shape[1])*.003;pen[0,0]=0
 info=x.T@((pr*(1-pr))[:,None]*x)/n;H=info+pen;hi=np.linalg.inv(H);meat=x.T@(((y-pr)**2)[:,None]*x)/(n*n);cov=hi@meat@hi
 s2=np.einsum('ij,ij->i',xv@cov,xv);se=.5*np.sqrt(np.maximum(s2,0));df=float(np.trace(hi@info));_,g=b.objective(coef,x,o,y,np.ones(n),.003);assert max(abs(g))<1e-4
 out.append(dict(fold=f,training_rows=n,positive_training_rows=int(sum(y)),parameters=x.shape[1],ridge_effective_degrees_of_freedom=df,positives_per_effective_parameter=float(sum(y)/df),conditional_shrunk_logit_SE_quantiles=np.quantile(se,[.1,.5,.9,.99]).tolist(),head_gradient_max=float(max(abs(g)))))
result=dict(status='complete',evaluation_labels_opened=False,folds=out,variance_estimator='Penalized logistic sandwich H^-1 X^Tdiag(residual^2)X H^-1/n^2; conditional on fixed backbone/design, then0.5headshrinkage.',limitations='Approximate conditional diagnostic. Does not establish causality, generalization improvement or independence; excludes context, feature-builder and selection uncertainty. No new fitted recipe or score.',hypothesis='If conditional estimation variance is material, honest inner crossfitting can expand calibration labels from6000 to56000. Every inner prior/encoder/context must fit without its heldout labels/covariates; final outer model refit preserves frozen outer evaluation.')
(R/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
