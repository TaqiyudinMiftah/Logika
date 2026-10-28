"""Three fixed feature-block contrasts; no performance is read until all folds fit."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,traceback,fcntl
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b,feature_head
from features import BLOCKS
R=Path(__file__).resolve().parents[1];A=R.parent
MAIN=A/'kumo-coverage-placement-20261021';DATA=A/'kumo-temperature-adaptation-20261021/full_data'
INPUT=A/'ranking-pipeline-20261020/input';HIST=A/'coverage-transfer-20261021'
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'plan.json')['original_deadline'] and not(R/'STOP').exists()
def emit(stage,**v):print(json.dumps(dict(stage=stage,time=time.time(),**v)),flush=True)
def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 n1=int(y.sum());auc=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1));assert abs(auc-m['AUC'])<1e-12
 m['logloss']=float(log_loss(y,p));return m
def reproduce():
 lock=read(INPUT/'BASELINE_LOCK.json');extra=read(HIST/'data/AUDIT.json')['rebuilt_V25_columns']
 columns=lock['correction']['features']+extra
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id']+lock['correction']['features'])
 coverage=pd.read_parquet(HIST/'data/features.parquet');assert original.application_id.tolist()==coverage.application_id.tolist()
 for c in extra:original[c]=coverage[c].to_numpy()
 checks=[]
 for f in range(5):
  ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');vi=ix['validation'];valid=original.iloc[vi]
  head=joblib.load(HIST/f'evidence/runs/V25/fold{f}_correction.joblib');assert head['columns']==columns
  raw=pd.read_parquet(INPUT/f'artifacts/fold{f}/kumo_oof.parquet');sp=pd.read_parquet(INPUT/f'artifacts/fold{f}/spline_oof.parquet')
  expected=pd.read_parquet(HIST/f'evidence/runs/V25/fold{f}_validation.parquet')
  assert raw.application_id.tolist()==sp.application_id.tolist()==expected.application_id.tolist()==valid.application_id.tolist()
  actual=.75*b.corrected_probability(head,valid,raw.prediction)[0]+.25*sp.prediction.to_numpy()
  error=float(np.max(abs(actual-expected.prediction.to_numpy())));assert error<1e-12
  checks.append(dict(fold=f,replay_error=error,rows=len(vi),model_sha256=b.sha(HIST/f'evidence/runs/V25/fold{f}_correction.joblib')))
 b.dump(dict(status='passed',checks=checks,outer_labels_opened=False,historical_reproduction_not_matched_prior_control=True),R/'BASELINE_REPRODUCTION.json')
 return lock
def main():
 flock=(R/'controller.lock').open('a+');fcntl.flock(flock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'TRAINING_COMPLETE.json').exists();plan=read(R/'plan.json');check()
 for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
 audit=read(R/'FEATURE_AUDIT.json');assert audit['target_labels_opened'] is False and audit['feature_shared_estimators'] is False
 assert b.sha(R/'features.parquet')==audit['feature_hash']==plan['features_sha256']
 independent=read(R/'INDEPENDENT_FEATURE_AUDIT.json');assert independent['status']=='passed'and independent['feature_hash']==audit['feature_hash']
 lock=reproduce();features=pd.read_parquet(R/'features.parquet').set_index('application_id');assert features.index.is_unique
 actual_lock=dict(Kumo=lock['Kumo'],correction={**lock['correction'],'features':lock['correction']['features']+read(HIST/'data/AUDIT.json')['rebuilt_V25_columns']},spline=lock['original_spline'],blend=[.75,.25],outer_fold_seed=31027,head_maxiter=500,head_ftol=1e-11,head_gtol=1e-7,baseline='Repaired V25: control90',changes_permitted='Correction input feature block only')
 b.dump(actual_lock,R/'MODEL_LOCK.json');checks=[]
 for f in range(5):
  d=DATA/f'fold{f}';man=read(d/'manifest.json')
  for n,h in man['files'].items():assert b.sha(d/n)==h
  ix=dict(np.load(d/'indices.npz'));q=pd.read_parquet(d/'query.parquet');y=np.load(d/'correction_labels.npy').astype(np.int64)
  context=pd.read_parquet(d/'context.parquet',columns=['application_id'])
  assert 'target'not in q and len(q)==20000 and len(y)==6000 and set(q.application_id).isdisjoint(context.application_id)
  training_rows=np.r_[ix['context'],ix['correction']]
  assert len(training_rows)==56000 and len(np.unique(training_rows))==56000
  assert not set(ix['validation'])&set(training_rows) and not set(ix['context'])&set(ix['correction'])
  original_ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz')
  assert np.array_equal(np.sort(training_rows),np.sort(original_ix['train']))
  for role in ['context','correction','validation']:assert np.array_equal(ix[role],original_ix[role])
  source=MAIN/f'conditions/fold{f}/control90';done=read(source/'complete.json')
  for n in ['raw_predictions.npz','correction.joblib','validation.npz']:assert b.sha(source/n)==done['small_files'][n]
  raw=np.load(source/'raw_predictions.npz');assert np.array_equal(raw['rows'],np.r_[ix['correction'],ix['validation']])
  old=joblib.load(source/'correction.joblib');assert old['columns']==actual_lock['correction']['features']
  x,o=b.design(old,q.iloc[:6000],raw['prediction'][:6000]);value,grad=b.objective(old['coefficients'],x,o,y,np.ones(6000),.003)
  assert abs(value-old['objective'])<1e-12 and max(abs(grad))<1e-4
  checks.append(dict(fold=f,head_objective_reproduced=True,outer_labels_opened=False,query_ids_sha256=b.fingerprint(q,['application_id'])))
 b.dump(dict(status='passed',checks=checks,new_features_target_free=True,source_hashes=plan['source_sha256'],feature_hash=audit['feature_hash']),R/'SCIENTIFIC_GATE.json')
 recipes={'baseline':[] ,**BLOCKS}
 for f in range(5):
  d=DATA/f'fold{f}';ix=dict(np.load(d/'indices.npz'));q=pd.read_parquet(d/'query.parquet');labels=np.load(d/'correction_labels.npy').astype(np.int64)
  source=MAIN/f'conditions/fold{f}/control90';raw=np.load(source/'raw_predictions.npz')['prediction'];old=joblib.load(source/'correction.joblib');sp=np.load(d/'spline_validation.npz');assert np.array_equal(sp['rows'],ix['validation'])
  feat=features.loc[q.application_id]
  for c in sum(BLOCKS.values(),[]):q[c]=feat[c].to_numpy(float)
  for name,extra in recipes.items():
   check();dest=R/f'folds/{name}/fold{f}'
   if (dest/'complete.json').exists():
    for n,h in read(dest/'complete.json')['files'].items():assert b.sha(dest/n)==h
    continue
   assert not dest.exists(),'Partial fit retained; explicit audit needed before resuming'
   dest.mkdir(parents=True);start=time.time()
   train=q.iloc[:6000].copy().assign(target=labels);valid=q.iloc[6000:].drop(columns=['target'],errors='ignore')
   head=feature_head.fit(train,raw[:6000],old['columns']+extra,.003,.5);joblib.dump(head,dest/'model.joblib',compress=3)
   prediction=.75*b.corrected_probability(head,valid,raw[6000:])[0]+.25*sp['prediction']
   replay=.75*b.corrected_probability(joblib.load(dest/'model.joblib'),valid,raw[6000:])[0]+.25*sp['prediction'];assert np.array_equal(prediction,replay)
   if name=='baseline':assert np.max(abs(prediction-np.load(source/'validation.npz')['prediction']))<2e-6
   np.savez_compressed(dest/'validation.npz',rows=ix['validation'],prediction=prediction,application_id=valid.application_id.to_numpy(str))
   b.dump(dict(fold=f,recipe=name,columns=head['columns'],head_rows=6000,outer_labels_opened=False,fit_seconds=time.time()-start,parameters=len(head['coefficients']),gradient_max_abs=head['gradient_max_abs'],prediction_replay_exact=True),dest/'boundary.json')
   b.dump(dict(status='complete',files={p.name:b.sha(p)for p in dest.iterdir()if p.is_file()}),dest/'complete.json');emit('fit_complete',fold=f,recipe=name)
 b.dump(dict(status='complete',fitted_recipes=4,heads=20,outer_labels_opened=False),R/'TRAINING_COMPLETE.json');report()
def report():
 assert read(R/'TRAINING_COMPLETE.json')['status']=='complete'
 z=np.load(MAIN/'oof.npz');y=z['target'].astype(np.int64);fold=z['fold'];pred={}
 for name in ['baseline']+list(BLOCKS):
  p=np.full(70000,np.nan);seen=np.zeros(70000,int)
  for f in range(5):
   d=R/f'folds/{name}/fold{f}'
   for n,h in read(d/'complete.json')['files'].items():assert b.sha(d/n)==h
   v=np.load(d/'validation.npz');ix=v['rows'];assert set(ix)==set(np.flatnonzero(fold==f));p[ix]=v['prediction'];seen[ix]+=1
  assert np.isfinite(p).all()and np.all(seen==1);pred[name]=p
 metrics={k:metric(y,p)for k,p in pred.items()};fm={k:[metric(y[fold==f],p[fold==f])for f in range(5)]for k,p in pred.items()}
 assert abs(metrics['baseline']['KS']-.5302185366909258)<2e-6
 effects={}
 for k in BLOCKS:
  delta={m:metrics[k][m]-metrics['baseline'][m]for m in ['KS','AUC','logloss']};fold_delta=[fm[k][f]['KS']-fm['baseline'][f]['KS']for f in range(5)]
  delta.update(fold_KS_deltas=fold_delta,KS_fold_wins=sum(x>0 for x in fold_delta),provisional_support=delta['KS']>0 and delta['AUC']>0 and sum(x>0 for x in fold_delta)>=3 and min(fold_delta)>=-.005);effects[k]=delta
 out=dict(status='point_metrics_complete',metrics=metrics,fold_metrics=fm,effects=effects,goal_point_reached=any(metrics[k]['KS']>=.54 for k in BLOCKS),historical_label_reuse=True,submission_generated=False)
 np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred);b.dump(out,R/'POINT_RESULT.json');emit('point_metrics_complete',metrics=metrics)
 rng=np.random.default_rng(20261116);groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]];bootstrap={k:[]for k in BLOCKS}
 for _ in range(1000):
  check();ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);base=b.scores(y[ix],pred['baseline'][ix])
  for k in BLOCKS:
   m=b.scores(y[ix],pred[k][ix]);bootstrap[k].append([m['KS']-base['KS'],m['AUC']-base['AUC']])
 for k in BLOCKS:effects[k]['simultaneous_conditional_interval']=np.quantile(bootstrap[k],[.05/12,1-.05/12],axis=0).tolist()
 out.update(status='complete',interval_scope='1000 paired fold/class bootstrap. Six comparisons (3 blocks x KS/AUC), Bonferroni95percent. Reused labels: excludes adaptive selection and retraining uncertainty.')
 b.dump(out,R/'result.json');np.savez_compressed(R/'bootstrap.npz',**bootstrap)
 lines=['# Fixed-model feature contrasts','', '| Feature block | KS | AUC | Delta KS | Delta AUC | KS fold wins |','|---|---:|---:|---:|---:|---:|']
 for k in BLOCKS:lines.append(f"| {k} | {metrics[k]['KS']:.8f} | {metrics[k]['AUC']:.8f} | {effects[k]['KS']:+.8f} | {effects[k]['AUC']:+.8f} | {effects[k]['KS_fold_wins']}/5 |")
 lines+=['','All20heads are retained. No model/regularization/blend search. These are correction-stage effects on historically reused development labels. No fresh confirmation or public improvement is claimed.','',f"Target .54 reached at point estimate: {out['goal_point_reached']}. Individual intervals and qualifications are in result.json."]
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');emit('complete',goal_point_reached=out['goal_point_reached'])
if __name__=='__main__':
 try:
  if len(sys.argv)>1 and sys.argv[1]=='report':report()
  else:main()
 except BaseException:
  b.dump(dict(status='failed',traceback=traceback.format_exc()),R/'FAILURE.json');raise
