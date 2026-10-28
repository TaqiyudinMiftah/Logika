"""Terminal evaluation only after every fold completes. No tuning or refitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import roc_auc_score,log_loss
import baseline as b
R=Path(__file__).resolve().parents[1]
def metric(y,p):
 m=b.scores(y,p);ind=ks_2samp(p[y==0],p[y==1]).statistic
 auc=(rankdata(p)[y==1].sum()-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(y==0))
 assert abs(m['KS']-ind)<1e-12 and abs(m['AUC']-auc)<1e-12
 m['logloss']=float(log_loss(y,p));return m

def main():
 assert json.loads((R/'state.json').read_text())['status']=='training_complete'
 y=np.full(70000,-1);ids=np.full(70000,'',dtype=object);folds=np.full(70000,-1);seen=np.zeros(70000,int)
 pred={v:np.full(70000,np.nan)for v in ['frozen','temperature']}
 for f in range(5):
  d=R/f'full_data/fold{f}';ix=dict(np.load(d/'indices.npz'));c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet')
  ci=ix['context'];known=y[ci]>=0;assert np.array_equal(y[ci][known],c.target.to_numpy()[known]);y[ci]=c.target.to_numpy();ids[ci]=c.application_id
  assert set(ci).isdisjoint(ix['correction'])and set(ci).isdisjoint(ix['validation'])and set(ix['correction']).isdisjoint(ix['validation'])
  complete=json.loads((R/f'runs/fold{f}/complete.json').read_text())
  for name,digest in complete['files'].items():assert b.sha(R/f'runs/fold{f}'/name)==digest
  vi=ix['validation'];seen[vi]+=1;folds[vi]=f
  for v in pred:
   z=dict(np.load(R/f'runs/fold{f}/{v}/validation.npz',allow_pickle=True));assert np.array_equal(z['rows'],vi)
   assert z['application_id'].tolist()==q.application_id.iloc[6000:].tolist();pred[v][vi]=z['prediction']
 assert np.all(seen==1)and np.all(y>=0)and len(set(ids))==70000
 pred['fixed_half_blend']=(pred['frozen']+pred['temperature'])/2
 assert all(np.isfinite(p).all()for p in pred.values())
 metrics={v:metric(y,p)for v,p in pred.items()};fm={v:[metric(y[folds==f],p[folds==f])for f in range(5)]for v,p in pred.items()}
 # Paired fold/class resampling, conditional on fitted OOF predictions.
 rng=np.random.default_rng(20261109);groups=[np.flatnonzero((folds==f)&(y==c))for f in range(5)for c in [0,1]];samples={v:[]for v in ['temperature','fixed_half_blend']}
 for i in range(2000):
  ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);base=b.scores(y[ix],pred['frozen'][ix])
  for v in samples:
   m=b.scores(y[ix],pred[v][ix]);samples[v].append([m['KS']-base['KS'],m['AUC']-base['AUC']])
 comparisons={v:dict(delta_KS=metrics[v]['KS']-metrics['frozen']['KS'],delta_AUC=metrics[v]['AUC']-metrics['frozen']['AUC'],
  conditional_intervals=np.quantile(samples[v],[.00625,.99375],axis=0).tolist(),fold_KS_deltas=[fm[v][f]['KS']-fm['frozen'][f]['KS']for f in range(5)])for v in samples}
 np.savez_compressed(R/'OOF.npz',target=y,application_id=np.asarray(ids,str),fold=folds,**pred)
 report=dict(status='complete',metrics=metrics,fold_metrics=fm,comparisons=comparisons,primary='temperature',
  interval_scope='2000 paired fold/class bootstrap, Bonferroni two recipes/two metrics; conditional on fitted predictions; historical search uncertainty excluded',
  goal_point_reached=metrics['temperature']['KS']>=.54,independent_KS_AUC_verified=True,full_OOF_rows=70000,
  historical_label_reuse=True,point_in_time_certified=False,submission_created=False,trained_recipe_count=1,hardware_control_repeat_count=1)
 b.dump(report,R/'result.json');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
