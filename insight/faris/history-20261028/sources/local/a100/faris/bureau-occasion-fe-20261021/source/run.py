import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,fcntl,traceback
import numpy as np,pandas as pd,joblib
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b,feature_head,spline
from build import EXTRA
R=Path(__file__).resolve().parents[1];A=R.parent;INPUT=A/'ranking-pipeline-20261020/input';DATA=A/'kumo-temperature-adaptation-20261021/full_data';MAIN=A/'kumo-coverage-placement-20261021';PRIOR=A/'repayment-severity-fe-20261021';CONTROL=A/'maturity-placement-fe-20261021'
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'plan.json')['original_deadline']and not(R/'STOP').exists()
def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12;n=int(y.sum());assert abs((rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n))-m['AUC'])<1e-12;m['logloss']=float(log_loss(y,p));return m
def main():
 lock=(R/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check();assert not(R/'TRAINING_COMPLETE.json').exists();plan=read(R/'plan.json')
 for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
 audit=read(R/'BUILD_COMPLETE.json');assert audit['default_labels_opened']is False and audit['source_sha256']==b.sha(R/'source/build.py')
 assert audit['measurement_source_sha256']==b.sha(R/'source/measurement.py')
 for receipt in audit['folds']:
  for n,key in [('features.parquet','feature_sha256'),('parameters.json','parameter_sha256')]:assert b.sha(R/f"fold{receipt['fold']}"/n)==receipt[key]
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id']);b.dump(dict(status='passed',static_bureau_reproduced=True,shared_measurement_screen_passed=True,default_labels_used_for_features=False,controls_same_width=True),R/'SCIENTIFIC_GATE.json')
 for f in range(5):
  check();packet=DATA/f'fold{f}';manifest=read(packet/'manifest.json')
  for n,h in manifest['files'].items():assert b.sha(packet/n)==h
  ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');q=pd.read_parquet(packet/'query.parquet');yc=np.load(packet/'correction_labels.npy').astype(np.int64);context=pd.read_parquet(packet/'context.parquet',columns=['application_id','target']);assert'target'not in q
  assert context.application_id.tolist()==original.application_id.iloc[ix['context']].tolist()and q.application_id.tolist()==original.application_id.iloc[np.r_[ix['correction'],ix['validation']]].tolist()
  labels=pd.concat([context.set_index('application_id').target,pd.Series(yc,index=q.application_id.iloc[:6000])]);assert labels.index.is_unique and len(labels)==56000 and set(labels.index).isdisjoint(q.application_id.iloc[6000:])
  source=MAIN/f'conditions/fold{f}/control90';assert b.sha(source/'raw_predictions.npz')==read(source/'complete.json')['small_files']['raw_predictions.npz'];raw=np.load(source/'raw_predictions.npz');assert np.array_equal(raw['rows'],np.r_[ix['correction'],ix['validation']]);raw=raw['prediction'];headcols=joblib.load(source/'correction.joblib')['columns'];assert len(headcols)==18
  oldsp=joblib.load(MAIN/f'shared/fold{f}/spline.joblib');fixed=A/f'fold-local-priors-20261021/fold{f}';assert b.sha(fixed/'features.parquet')==read(fixed/'complete.json')['feature_table_sha256'];frame=pd.read_parquet(fixed/'features.parquet',columns=['application_id']+oldsp['columns']).set_index('application_id').loc[original.application_id].reset_index()
  for c in oldsp['categorical']:frame[c]=frame[c].astype(object)
  prior=PRIOR/f'fold{f}';assert b.sha(prior/'features.parquet')==read(prior/'transformer.json')['feature_hash'];means=pd.read_parquet(prior/'features.parquet').set_index('application_id')
  for t in [1,30]:q[f're_late_{t}_posterior']=means.loc[q.application_id,f'matched_beta_late_{t}'].to_numpy(float);frame[f're_late_{t}_posterior']=means.loc[frame.application_id,f'matched_beta_late_{t}'].to_numpy(float)
  oldh=joblib.load(PRIOR/f'heads/matched_maturity/fold{f}/model.joblib');olds=joblib.load(CONTROL/f'fold{f}/spline.joblib');before=.75*b.corrected_probability(oldh,q.iloc[6000:],raw[6000:])[0]+.25*b.predict_spline(olds,frame.iloc[ix['validation']]);saved=np.load(CONTROL/f'fold{f}/validation.npz');assert np.array_equal(saved['rows'],ix['validation'])and np.max(abs(before-saved['both']))<2e-14
  for kind in ['occasion']:
   features={kind:pd.read_parquet(R/f'fold{f}/features.parquet').set_index('application_id')}
   check();dest=R/f'models/{kind}/fold{f}';assert not dest.exists();dest.mkdir(parents=True);start=time.time();query=q.copy();table=frame.copy()
   for c in EXTRA:query[c]=features[kind].loc[query.application_id,c].to_numpy(float);table[c]=features[kind].loc[table.application_id,c].to_numpy(float)
   head=feature_head.fit(query.iloc[:6000].assign(target=yc),raw[:6000],headcols,.003,.5);training=table.iloc[ix['train']].copy();training['target']=labels.loc[training.application_id].to_numpy(np.int64);sp=spline.fit(training,oldsp['columns']);assert len(head['columns'])==18 and len(sp['columns'])==77 and sp['resolved_parameters']==olds['resolved_parameters']
   joblib.dump(head,dest/'head.joblib',compress=3);joblib.dump(sp,dest/'spline.joblib',compress=3)
   def predict(h,s):return .75*b.corrected_probability(h,query.iloc[6000:],raw[6000:])[0]+.25*b.predict_spline(s,table.iloc[ix['validation']])
   pr=predict(head,sp);replayed=predict(joblib.load(dest/'head.joblib'),joblib.load(dest/'spline.joblib'));assert np.array_equal(pr,replayed)
   np.savez_compressed(dest/'validation.npz',rows=ix['validation'],application_id=query.application_id.iloc[6000:].to_numpy(str),prediction=pr)
   b.dump(dict(fold=f,variant=kind,head_rows=6000,spline_rows=56000,head_features=18,spline_features=77,head_gradient_max_abs=head['gradient_max_abs'],head_iterations=head['iterations'],seconds=time.time()-start,feature_hash=audit['folds'][f]['feature_sha256'],default_validation_labels_opened=False,original_hyperparameters_unchanged=True,model_replay_exact=True),dest/'boundary.json');b.dump(dict(status='complete',files={p.name:b.sha(p)for p in dest.iterdir()if p.is_file()}),dest/'complete.json');print(json.dumps(dict(stage='model_fold_complete',fold=f,variant=kind)),flush=True)
 b.dump(dict(status='complete',folds=5,models=10,default_validation_labels_opened=False),R/'TRAINING_COMPLETE.json');report()
def report():
 assert read(R/'TRAINING_COMPLETE.json')['status']=='complete';z=np.load(CONTROL/'OOF.npz');y=z['target'].astype(np.int64);fold=z['fold'];pred=dict(baseline=z['baseline'],maturity=z['both'])
 for kind in ['occasion']:
  pr=np.full(70000,np.nan);seen=np.zeros(70000,int)
  for f in range(5):
   root=R/f'models/{kind}/fold{f}'
   for n,h in read(root/'complete.json')['files'].items():assert b.sha(root/n)==h
   a=np.load(root/'validation.npz');rows=a['rows'];assert set(rows)==set(np.flatnonzero(fold==f));pr[rows]=a['prediction'];seen[rows]+=1
  assert np.all(seen==1)and np.isfinite(pr).all();pred[kind]=pr
 mm={k:metric(y,p)for k,p in pred.items()};fm={k:[metric(y[fold==f],p[fold==f])for f in range(5)]for k,p in pred.items()};out=dict(status='point_metrics_complete',metrics=mm,fold_metrics=fm,goal_point_reached=mm['occasion']['KS']>=.54,confirmation_opened=False,historical_label_reuse=True,submission_generated=False)
 np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred);b.dump(out,R/'POINT_RESULT.json');print(json.dumps(dict(stage='point_metrics_complete',metrics=mm)),flush=True)
 rng=np.random.default_rng(20261203);groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]];boot={k:[]for k in ['baseline','maturity']}
 for _ in range(1000):
  check();rows=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);m={k:b.scores(y[rows],p[rows])for k,p in pred.items()}
  for k in boot:boot[k].append([m['occasion']['KS']-m[k]['KS'],m['occasion']['AUC']-m[k]['AUC']])
 out.update(status='complete',intervals={k:np.quantile(v,[.05/8,1-.05/8],axis=0).tolist()for k,v in boot.items()},uncertainty='1000pairedfold/classbootstrap,2contrasts x2metricsBonferroni95percent; conditional,notcorrectedforhistorical/adaptiveselection/retraininguncertainty.');b.dump(out,R/'result.json');np.savez_compressed(R/'bootstrap.npz',**boot)
if __name__=='__main__':
 try:main()
 except BaseException:b.dump(dict(status='failed',error=traceback.format_exc()),R/'FAILURE.json');raise
