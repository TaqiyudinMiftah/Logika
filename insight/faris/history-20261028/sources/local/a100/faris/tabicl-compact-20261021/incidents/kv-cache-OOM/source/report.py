"""All-fold terminal evaluator; no learned ensemble or threshold search."""
from pathlib import Path
import numpy as np,json
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
R=Path(__file__).resolve().parents[1];M=R.parent/'kumo-coverage-placement-20261021'
def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 a=(rankdata(p)[y==1].sum()-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(y==0));assert abs(a-m['AUC'])<1e-12
 m['logloss']=float(log_loss(y,p));return m
z=np.load(M/'oof.npz');y=z['target'];folds=z['fold'];p=np.full(len(y),np.nan);seen=np.zeros(len(y),int);checks=[]
for f in range(5):
 d=R/f'folds/fold{f}';c=json.loads((d/'complete.json').read_text())
 for name,digest in c['files'].items():assert b.sha(d/name)==digest
 v=np.load(d/'validation.npz');assert np.array_equal(v['rows'],np.flatnonzero(folds==f))or set(v['rows'])==set(np.flatnonzero(folds==f))
 p[v['rows']]=v['prediction'];seen[v['rows']]+=1;checks.append(c)
assert np.all(seen==1)and np.isfinite(p).all()
pred={'control90':z['control90'],'tabicl90':p,'fixed20percent':.8*z['control90']+.2*p};metrics={k:metric(y,v)for k,v in pred.items()};fm={k:[metric(y[folds==f],v[folds==f])for f in range(5)]for k,v in pred.items()}
rng=np.random.default_rng(20261111);groups=[np.flatnonzero((folds==f)&(y==c))for f in range(5)for c in [0,1]];boot={k:[]for k in ['tabicl90','fixed20percent']}
for i in range(2000):
 ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);base=b.scores(y[ix],pred['control90'][ix])
 for k in boot:
  m=b.scores(y[ix],pred[k][ix]);boot[k].append([m['KS']-base['KS'],m['AUC']-base['AUC']])
np.savez_compressed(R/'OOF.npz',target=y,fold=folds,**pred)
b.dump(dict(status='complete',metrics=metrics,fold_metrics=fm,conditional_intervals={k:np.quantile(v,[.00625,.99375],axis=0).tolist()for k,v in boot.items()},
 goal_point_reached=metrics['tabicl90']['KS']>=.54 or metrics['fixed20percent']['KS']>=.54,
 interval_scope='2000 paired fold/class bootstrap, two recipes/two metrics Bonferroni; excludes historical adaptive search and retraining uncertainty',
 full_OOF_rows=70000,fitted_pipeline_count=1,fixed_ensemble_count=1,historical_label_reuse=True,submission_generated=False),R/'result.json')
