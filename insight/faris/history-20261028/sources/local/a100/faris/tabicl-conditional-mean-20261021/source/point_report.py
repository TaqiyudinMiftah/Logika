from pathlib import Path
import numpy as np,json
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
R=Path(__file__).resolve().parents[1];z=np.load(R/'reference.npz');y=z['target'];fold=z['fold'];p=np.full(70000,np.nan);seen=np.zeros(70000,int)
for f in range(5):
 d=R/f'folds/fold{f}';c=json.loads((d/'complete.json').read_text())
 for n,h in c['files'].items():assert b.sha(d/n)==h
 v=np.load(d/'validation.npz');ix=v['rows'];assert set(ix)==set(np.flatnonzero(fold==f));p[ix]=v['prediction'];seen[ix]+=1
assert np.all(seen==1)and np.isfinite(p).all()
def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 a=(rankdata(p)[y==1].sum()-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(y==0));assert abs(a-m['AUC'])<1e-12
 m['logloss']=float(log_loss(y,p));return m
pred=dict(control90=z['control90'],tabicl90=p,fixed20percent=.8*z['control90']+.2*p)
np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred)
b.dump(dict(status='complete_point_verification',metrics={k:metric(y,v)for k,v in pred.items()},fold_metrics={k:[metric(y[fold==f],v[fold==f])for f in range(5)]for k,v in pred.items()},full_OOF_rows=70000,verification_repeat=False,model='TabICLv2_regression_conditional_mean',cache_mode='kv',device='A100',uncertainty='Paired uncertainty to be computed on PC2; not a promotion.',goal_point_reached=any(metric(y,v)['KS']>=.54 for v in pred.values()),submission_generated=False),R/'result.json')
