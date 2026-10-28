from pathlib import Path
import json,numpy as np
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
R=Path(__file__).resolve().parents[1];z=np.load(R/'reference.npz');y=z['target'];fold=z['fold'];pred={'control90':z['control90']};complete=[];partial={}
for view in ['full','compact']:
 missing=[f for f in range(5)if not(R/f'runs/{view}/folds/fold{f}/complete.json').exists()]
 if missing:partial[view]=missing;continue
 p=np.full(70000,np.nan);cap=np.full(70000,np.nan);seen=np.zeros(70000,int)
 for f in range(5):
  d=R/f'runs/{view}/folds/fold{f}';c=json.loads((d/'complete.json').read_text())
  for n,h in c['files'].items():assert b.sha(d/n)==h
  v=np.load(d/'validation.npz');assert set(v['rows'])==set(np.flatnonzero(fold==f));p[v['rows']]=v['prediction'];cap[v['rows']]=v['matched_300_prediction'];seen[v['rows']]+=1
 assert np.all(seen==1)and np.isfinite(p).all();pred[view]=p;pred[view+'_cap300']=cap;pred[view+'_fixed20']=.8*z['control90']+.2*p;complete.append(view)
assert complete

def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 a=(rankdata(p)[y==1].sum()-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(y==0));assert abs(a-m['AUC'])<1e-12
 m['logloss']=float(log_loss(y,p));m['Brier']=float(np.mean((p-y)**2));return m
metrics={k:metric(y,p)for k,p in pred.items()};np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred)
result=dict(status='point_metrics_complete',metrics=metrics,fold_metrics={k:[metric(y[fold==f],v[fold==f])for f in range(5)]for k,v in pred.items()},completed_views=complete,partial_views=partial,goal_point_reached=any(m['KS']>=.54 for k,m in metrics.items()if k!='control90'),full_OOF_rows=70000,hyperparameter_grid=False,outer_labels_used_for_stopping=False,trained_pipeline_count=len(complete),derived_prefix_diagnostic_count=len(complete),historical_label_reuse=True,submission_generated=False)
b.dump(result,R/'POINT_RESULT.json')
rng=np.random.default_rng(20261112);groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]];boot={k:[]for k in pred if k!='control90'}
for i in range(2000):
 ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);base=b.scores(y[ix],pred['control90'][ix])
 for k in boot:
  m=b.scores(y[ix],pred[k][ix]);boot[k].append([m['KS']-base['KS'],m['AUC']-base['AUC']])
result.update(status='complete',conditional_intervals={k:np.quantile(v,[.0020833333333333333,.9979166666666667],axis=0).tolist()for k,v in boot.items()},interval_scope='2000 paired fold/class bootstrap; six prespecified views/two metrics Bonferroni. Prefixes share training trajectories. Conditional on predictions, excludes historical search/retraining uncertainty.')
b.dump(result,R/'result.json')
