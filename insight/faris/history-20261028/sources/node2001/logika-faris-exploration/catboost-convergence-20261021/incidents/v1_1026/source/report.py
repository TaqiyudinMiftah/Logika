from pathlib import Path
import json,numpy as np
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
R=Path(__file__).resolve().parents[1];z=np.load(R/'reference.npz');y=z['target'];fold=z['fold'];p=np.full(70000,np.nan);cap=np.full(70000,np.nan);seen=np.zeros(70000,int)
for f in range(5):
 d=R/f'folds/fold{f}';c=json.loads((d/'complete.json').read_text())
 for n,h in c['files'].items():assert b.sha(d/n)==h
 v=np.load(d/'validation.npz');assert set(v['rows'])==set(np.flatnonzero(fold==f));p[v['rows']]=v['prediction'];cap[v['rows']]=v['matched_300_prediction'];seen[v['rows']]+=1
assert np.all(seen==1)and np.isfinite(p).all()
def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 a=(rankdata(p)[y==1].sum()-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(y==0));assert abs(a-m['AUC'])<1e-12
 m['logloss']=float(log_loss(y,p));m['Brier']=float(np.mean((p-y)**2));return m
pred=dict(control90=z['control90'],catboost=p,matched_300_cap=cap,fixed20percent=.8*z['control90']+.2*p);metrics={k:metric(y,v)for k,v in pred.items()}
np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred)
b.dump(dict(status='point_metrics_complete',metrics=metrics,fold_metrics={k:[metric(y[fold==f],v[fold==f])for f in range(5)]for k,v in pred.items()},goal_point_reached=any(metrics[k]['KS']>=.54 for k in ['catboost','matched_300_cap','fixed20percent']),full_OOF_rows=70000,hyperparameter_grid=False,outer_labels_used_for_stopping=False,historical_label_reuse=True,submission_generated=False),R/'POINT_RESULT.json')
rng=np.random.default_rng(20261112);groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]];boot={k:[]for k in ['catboost','matched_300_cap','fixed20percent']}
for i in range(2000):
 ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);base=b.scores(y[ix],pred['control90'][ix])
 for k in boot:
  m=b.scores(y[ix],pred[k][ix]);boot[k].append([m['KS']-base['KS'],m['AUC']-base['AUC']])
result=json.loads((R/'POINT_RESULT.json').read_text());result.update(status='complete',conditional_intervals={k:np.quantile(v,[.004166666666666667,.9958333333333333],axis=0).tolist()for k,v in boot.items()},interval_scope='2000 paired fold/class bootstrap, three prespecified views/two metrics Bonferroni; matched300cap is a shared-trajectory diagnostic, not an independent training run. Conditional on fitted predictions; not a correction for historical research selection.')
b.dump(result,R/'result.json')
