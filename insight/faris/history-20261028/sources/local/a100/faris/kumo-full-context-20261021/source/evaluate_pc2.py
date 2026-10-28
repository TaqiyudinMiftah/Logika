"""Terminal full-OOF metrics, separate from the A100 training worker."""
from pathlib import Path
import json,hashlib,numpy as np
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
R=Path(__file__).resolve().parents[1];z=np.load(R.parent/'kumo-coverage-placement-20261021/oof.npz');y=z['target'];fold=z['fold'];seen=np.zeros(70000,int);pred={k:np.full(70000,np.nan)for k in ['context50000','context56000']}
for f in range(5):
 d=R/f'folds/fold{f}';c=json.loads((d/'complete.json').read_text());assert c['status']=='complete'
 for n,h in c['files'].items():assert b.sha(d/n)==h
 v=np.load(d/'validation.npz');assert set(v['rows'])==set(np.flatnonzero(fold==f));seen[v['rows']]+=1
 for k in pred:pred[k][v['rows']]=v[k]
assert np.all(seen==1)
pred['fixed_half']=.5*pred['context50000']+.5*pred['context56000'];pred['original_control90']=z['control90']
def metric(y,p):
 assert np.isfinite(p).all();m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 a=(rankdata(p)[y==1].sum()-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(y==0));assert abs(m['AUC']-a)<1e-12;m['logloss']=float(log_loss(y,p));return m
result=dict(status='point_metrics_complete',metrics={k:metric(y,p)for k,p in pred.items()},fold_metrics={k:[metric(y[fold==f],p[fold==f])for f in range(5)]for k,p in pred.items()},rows=70000,historical_label_reuse=True,goal_point_reached=any(metric(y,p)['KS']>=.54 for p in pred.values()),submission_generated=False)
np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred);b.dump(result,R/'POINT_RESULT.json')
rng=np.random.default_rng(20261113);groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]];boot={k:[]for k in ['context56000','fixed_half']}
for i in range(2000):
 ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);base=metric(y[ix],pred['context50000'][ix])
 for k in boot:
  m=b.scores(y[ix],pred[k][ix]);boot[k].append([m['KS']-base['KS'],m['AUC']-base['AUC']])
result.update(status='complete',paired_intervals={k:np.quantile(v,[.00625,.99375],axis=0).tolist()for k,v in boot.items()},interval_scope='Fold/class stratified2000paired bootstrap, two frozen contrasts and two metrics Bonferroni; conditional on predictions, not historical search or retraining uncertainty.')
b.dump(result,R/'result.json')
