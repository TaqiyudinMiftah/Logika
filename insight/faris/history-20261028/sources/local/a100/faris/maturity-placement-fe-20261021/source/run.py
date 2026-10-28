import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,fcntl,traceback
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b,spline
R=Path(__file__).resolve().parents[1];A=R.parent;MAIN=A/'kumo-coverage-placement-20261021';DATA=A/'kumo-temperature-adaptation-20261021/full_data';INPUT=A/'ranking-pipeline-20261020/input';PRIOR=A/'repayment-severity-fe-20261021';FIX=A/'fold-local-priors-20261021'
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'plan.json')['original_deadline']and not(R/'STOP').exists()
def emit(stage,**v):print(json.dumps(dict(stage=stage,time=time.time(),**v)),flush=True)
def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 n=int(y.sum());assert abs((rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n))-m['AUC'])<1e-12;m['logloss']=float(log_loss(y,p));return m
def main():
 lock=(R/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check();plan=read(R/'plan.json');assert not(R/'TRAINING_COMPLETE.json').exists()
 for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
 assert read(PRIOR/'TRAINING_COMPLETE.json')['status']=='complete';assert read(PRIOR/'SCIENTIFIC_GATE.json')['status']=='passed'
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);allchecks=[]
 for f in range(5):
  check();ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');d=DATA/f'fold{f}';packet=read(d/'manifest.json')
  for n,h in packet['files'].items():assert b.sha(d/n)==h
  oldroot=MAIN/f'shared/fold{f}';oldcomplete=read(oldroot/'complete.json')
  for n,h in oldcomplete['files'].items():assert b.sha(oldroot/n)==h
  old=joblib.load(oldroot/'spline.joblib');assert len(old['columns'])==77
  repaired=FIX/f'fold{f}';assert b.sha(repaired/'features.parquet')==read(repaired/'complete.json')['feature_table_sha256']
  frame=pd.read_parquet(repaired/'features.parquet',columns=['application_id']+old['columns']).set_index('application_id').loc[original.application_id].reset_index()
  for c in old['categorical']:frame[c]=frame[c].astype(object)
  assert'target'not in frame and frame.application_id.tolist()==original.application_id.tolist()
  valid=frame.iloc[ix['validation']].copy();previous=b.predict_spline(old,valid);sp=np.load(oldroot/'validation.npz');assert np.array_equal(sp['rows'],ix['validation']);assert np.max(abs(previous-sp['prediction']))<2e-14
  context=pd.read_parquet(d/'context.parquet',columns=['application_id','target']);q=pd.read_parquet(d/'query.parquet');yc=np.load(d/'correction_labels.npy').astype(np.int64)
  assert'target'not in q and context.application_id.tolist()==original.application_id.iloc[ix['context']].tolist()and q.application_id.iloc[:6000].tolist()==original.application_id.iloc[ix['correction']].tolist()
  assert q.application_id.iloc[6000:].tolist()==original.application_id.iloc[ix['validation']].tolist()
  labels=pd.concat([context.set_index('application_id').target,pd.Series(yc,index=q.application_id.iloc[:6000])]);assert labels.index.is_unique and len(labels)==56000
  assert set(labels.index)==set(original.application_id.iloc[ix['train']])and set(labels.index).isdisjoint(valid.application_id)
  feats=PRIOR/f'fold{f}';pp=read(feats/'transformer.json');assert b.sha(feats/'features.parquet')==pp['feature_hash'];replacement=pd.read_parquet(feats/'features.parquet').set_index('application_id').loc[frame.application_id]
  for t in [1,30]:frame[f're_late_{t}_posterior']=replacement[f'matched_beta_late_{t}'].to_numpy(float)
  training=frame.iloc[ix['train']].copy();training['target']=labels.loc[training.application_id].to_numpy(np.int64);valid=frame.iloc[ix['validation']].copy();assert'target'not in valid
  dest=R/f'fold{f}';assert not dest.exists();dest.mkdir();start=time.time();model=spline.fit(training,old['columns'])
  assert model['resolved_parameters']==old['resolved_parameters'];assert model['model'].coef_.shape==old['model'].coef_.shape
  joblib.dump(model,dest/'spline.joblib',compress=3);newsp=b.predict_spline(model,valid);replay=b.predict_spline(joblib.load(dest/'spline.joblib'),valid);assert np.array_equal(newsp,replay)
  source=MAIN/f'conditions/fold{f}/control90';raw=np.load(source/'raw_predictions.npz');assert np.array_equal(raw['rows'],np.r_[ix['correction'],ix['validation']]);raw=raw['prediction'][6000:]
  heads={};oldq=q.iloc[6000:].copy();newq=oldq.copy()
  for t in [1,30]:newq[f're_late_{t}_posterior']=replacement[f'matched_beta_late_{t}'].to_numpy()[ix['validation']]
  for name,query in [('baseline',oldq),('matched_maturity',newq)]:
   root=PRIOR/f'heads/{name}/fold{f}';c=read(root/'complete.json')
   for n,h in c['files'].items():assert b.sha(root/n)==h
   heads[name]=b.corrected_probability(joblib.load(root/'model.joblib'),query,raw)[0]
   expected=np.load(root/'validation.npz')['prediction'];assert np.max(abs(.75*heads[name]+.25*previous-expected))<2e-14
  preds=dict(baseline=.75*heads['baseline']+.25*previous,head_only=.75*heads['matched_maturity']+.25*previous,spline_only=.75*heads['baseline']+.25*newsp,both=.75*heads['matched_maturity']+.25*newsp)
  np.savez_compressed(dest/'validation.npz',rows=ix['validation'],application_id=valid.application_id.to_numpy(str),**preds)
  boundary=dict(fold=f,fit_rows=56000,prior_fit_rows=50000,query_labels_opened=False,source_prior_hash=b.sha(feats/'transformer.json'),old_spline_replay_max_error=float(abs(previous-sp['prediction']).max()),new_spline_replay_exact=True,parameter_configuration_unchanged=True,design_width_unchanged=True,seconds=time.time()-start)
  b.dump(boundary,dest/'boundary.json');b.dump(dict(status='complete',files={z.name:b.sha(z)for z in dest.iterdir()if z.is_file()}),dest/'complete.json');allchecks.append(boundary);emit('spline_complete',fold=f)
 b.dump(dict(status='complete',folds=allchecks,models=5,outer_labels_opened=False),R/'TRAINING_COMPLETE.json');report()
def report():
 assert read(R/'TRAINING_COMPLETE.json')['status']=='complete';z=np.load(MAIN/'oof.npz');y=z['target'].astype(np.int64);fold=z['fold'];pred={}
 for f in range(5):
  root=R/f'fold{f}'
  for n,h in read(root/'complete.json')['files'].items():assert b.sha(root/n)==h
  d=dict(np.load(root/'validation.npz'));rows=d.pop('rows');d.pop('application_id');assert set(rows)==set(np.flatnonzero(fold==f))
  for k,p in d.items():
   if k not in pred:pred[k]=np.full(70000,np.nan)
   pred[k][rows]=p
 assert all(np.isfinite(p).all()for p in pred.values());mm={k:metric(y,p)for k,p in pred.items()};fm={k:[metric(y[fold==f],p[fold==f])for f in range(5)]for k,p in pred.items()}
 pairs={'both_vs_baseline':('both','baseline'),'both_vs_head_only':('both','head_only'),'spline_vs_baseline':('spline_only','baseline')}
 effects={k:dict(KS=mm[a]['KS']-mm[c]['KS'],AUC=mm[a]['AUC']-mm[c]['AUC'],fold_KS_deltas=[fm[a][f]['KS']-fm[c][f]['KS']for f in range(5)])for k,(a,c)in pairs.items()}
 out=dict(status='point_metrics_complete',metrics=mm,fold_metrics=fm,effects=effects,goal_point_reached=mm['both']['KS']>=.54,confirmation_opened=False,historical_label_reuse=True,submission_generated=False)
 np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred);b.dump(out,R/'POINT_RESULT.json');emit('point_metrics_complete',metrics=mm)
 rng=np.random.default_rng(20261118);groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]];boot={k:[]for k in pairs}
 for _ in range(1000):
  check();ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);m={k:b.scores(y[ix],p[ix])for k,p in pred.items()}
  for k,(a,c)in pairs.items():boot[k].append([m[a]['KS']-m[c]['KS'],m[a]['AUC']-m[c]['AUC']])
 out.update(status='complete',intervals={k:np.quantile(v,[.05/12,1-.05/12],axis=0).tolist()for k,v in boot.items()},uncertainty='1000 paired fold/class bootstrap,3contrasts x2metrics,Bonferroni95percent. Adaptive feature selection and historical label reuse are not corrected. Conditional, not independent confirmation.')
 b.dump(out,R/'result.json');np.savez_compressed(R/'bootstrap.npz',**boot);emit('complete',goal_point_reached=out['goal_point_reached'])
if __name__=='__main__':
 try:main()
 except BaseException:b.dump(dict(status='failed',error=traceback.format_exc()),R/'FAILURE.json');raise
