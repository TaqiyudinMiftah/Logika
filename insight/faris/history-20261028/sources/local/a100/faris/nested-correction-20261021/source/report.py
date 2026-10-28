from pathlib import Path
import json,numpy as np
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
R=Path(__file__).resolve().parents[1];assert json.loads((R/'HEADS_COMPLETE.json').read_text())['status']=='complete'
z=np.load(R.parent/'kumo-coverage-placement-20261021/oof.npz');y=z['target'].astype(np.int64);fold=z['fold'];pred={k:np.full(70000,np.nan)for k in ['crossfit56000','heldout6000']};seen=np.zeros(70000,int);replayed={k:np.full(70000,np.nan)for k in ['crossfit56000','heldout6000']}
for f in range(5):
 d=R/f'heads/fold{f}';c=json.loads((d/'complete.json').read_text());assert c['status']=='complete'
 for n,h in c['files'].items():assert b.sha(d/n)==h
 v=np.load(d/'validation.npz');assert set(v['rows'])==set(np.flatnonzero(fold==f));seen[v['rows']]+=1
 for k in pred:pred[k][v['rows']]=v[k];replayed[k][v['rows']]=v['replayed_'+k]
assert np.all(seen==1)
replayed['fixed_half']=.5*replayed['crossfit56000']+.5*replayed['heldout6000']
pred['fixed_half']=.5*pred['crossfit56000']+.5*pred['heldout6000'];pred['original_control90']=z['control90']
def metric(y,p):
 assert np.isfinite(p).all();m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 n1=int(np.sum(y));n0=len(y)-n1;a=float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0));assert abs(a-m['AUC'])<1e-12;m['logloss']=float(log_loss(y,p));return m
result=dict(status='point_metrics_complete',metrics={k:metric(y,p)for k,p in pred.items()},fold_metrics={k:[metric(y[fold==f],p[fold==f])for f in range(5)]for k,p in pred.items()},serialized_model_replay_metrics={k:metric(y,p)for k,p in replayed.items()},rows=70000,outer_folds=5,inner_folds=5,goal_point_reached=any(metric(y,pred[k])['KS']>=.54 and metric(y,replayed[k])['KS']>=.54 for k in ['crossfit56000','heldout6000','fixed_half']),historical_label_reuse=True,submission_generated=False)
np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred);np.savez_compressed(R/'MODEL_REPLAY_OOF.npz',target=y,fold=fold,**replayed);b.dump(result,R/'POINT_RESULT.json')
rng=np.random.default_rng(20261114);groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in[0,1]];contrasts=[('crossfit56000','heldout6000'),('crossfit56000','original_control90'),('fixed_half','original_control90')];boot={a+'_vs_'+b:[]for a,b in contrasts}
for i in range(2000):
 ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);m={k:b.scores(y[ix],p[ix])for k,p in pred.items()}
 for a,c in contrasts:boot[a+'_vs_'+c].append([m[a]['KS']-m[c]['KS'],m[a]['AUC']-m[c]['AUC']])
result.update(status='complete',conditional_intervals={k:np.quantile(v,[.004166666666666667,.9958333333333333],axis=0).tolist()for k,v in boot.items()},interval_scope='2000paired fold/class bootstrap;3frozen contrasts2metrics Bonferroni. Conditional on predictions; does not correct historical search or retraining uncertainty.')
b.dump(result,R/'result.json')
