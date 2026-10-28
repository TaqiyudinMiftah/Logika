import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,fcntl,traceback
import numpy as np,pandas as pd,joblib
from scipy.stats import ks_2samp,rankdata,betabinom
from scipy.special import betaln,gammaln
import baseline as b,feature_head,spline
R=Path(__file__).resolve().parents[1];A=R.parent;INPUT=A/'ranking-pipeline-20261020/input';MAIN=A/'kumo-coverage-placement-20261021';DATA=A/'kumo-temperature-adaptation-20261021/full_data';FIX=A/'fold-local-priors-20261021';CONTROL=A/'maturity-placement-fe-20261021'
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'plan.json')['original_deadline']and not(R/'STOP').exists()
def emit(stage,**v):print(json.dumps(dict(stage=stage,time=time.time(),**v)),flush=True)
def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12;n=int(y.sum());assert abs((rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n))-m['AUC'])<1e-12;return m
def main():
 lock=(R/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check();plan=read(R/'plan.json');assert not(R/'TRAINING_COMPLETE.json').exists()
 for n,h in plan['source_sha256'].items():assert b.sha(R/'source'/n)==h
 audit=read(R/'BUILD_COMPLETE.json');assert audit['default_labels_opened']is False and audit['source_sha256']==b.sha(R/'source/build.py')
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);gate=[]
 for f in range(5):
  tx=read(R/f'fold{f}/transformer.json');assert b.sha(R/f'fold{f}/features.parquet')==tx['feature_hash']
  for spec in tx['thresholds']:
   assert spec['quadrature_128_vs_256_max_error']<1e-6 and spec['iid_analytical_max_error']<1e-9
   ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');assert spec['fit_ids_sha256']==b.ids_hash(original,ix['context'])
   # Independent library check of the per-loan beta-binomial likelihood.
   p=np.array([.01,.1,.5,.9,.99]);k=np.array([0,1,4,8,12]);n=np.full(5,12);kap=spec['loan_moments']['kappa'];aa=kap*p;bb=kap*(1-p);direct=betaln(k+aa,n-k+bb)-betaln(aa,bb)+gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
   assert np.max(abs(direct-betabinom.logpmf(k,n,aa,bb)))<1e-12
  gate.append(dict(fold=f,context_fit_ids_verified=True,independent_beta_binomial_formula=True,analytical_iid_and_quadrature_reconstruction=True))
 b.dump(dict(status='passed',folds=gate,prior_default_labels_opened=False),R/'SCIENTIFIC_GATE.json')
 for f in range(5):
  check();d=DATA/f'fold{f}';manifest=read(d/'manifest.json')
  for n,h in manifest['files'].items():assert b.sha(d/n)==h
  ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');q=pd.read_parquet(d/'query.parquet');yc=np.load(d/'correction_labels.npy').astype(np.int64);ctx=pd.read_parquet(d/'context.parquet',columns=['application_id','target']);assert'target'not in q
  assert ctx.application_id.tolist()==original.application_id.iloc[ix['context']].tolist()and q.application_id.tolist()==original.application_id.iloc[np.r_[ix['correction'],ix['validation']]].tolist()
  labels=pd.concat([ctx.set_index('application_id').target,pd.Series(yc,index=q.application_id.iloc[:6000])]);assert labels.index.is_unique and len(labels)==56000 and set(labels.index).isdisjoint(q.application_id.iloc[6000:])
  source=MAIN/f'conditions/fold{f}/control90';done=read(source/'complete.json')
  for n in ['raw_predictions.npz','correction.joblib']:assert b.sha(source/n)==done['small_files'][n]
  raw=np.load(source/'raw_predictions.npz');assert np.array_equal(raw['rows'],np.r_[ix['correction'],ix['validation']]);raw=raw['prediction'];oldhead=joblib.load(source/'correction.joblib')
  ss=MAIN/f'shared/fold{f}';sc=read(ss/'complete.json')
  for n,h in sc['files'].items():assert b.sha(ss/n)==h
  oldsp=joblib.load(ss/'spline.joblib');fixed=FIX/f'fold{f}';assert b.sha(fixed/'features.parquet')==read(fixed/'complete.json')['feature_table_sha256']
  frame=pd.read_parquet(fixed/'features.parquet',columns=['application_id']+oldsp['columns']).set_index('application_id').loc[original.application_id].reset_index()
  for c in oldsp['categorical']:frame[c]=frame[c].astype(object)
  previous=b.predict_spline(oldsp,frame.iloc[ix['validation']]);expected=np.load(ss/'validation.npz');assert np.array_equal(expected['rows'],ix['validation'])and np.max(abs(previous-expected['prediction']))<2e-14
  feature=pd.read_parquet(R/f'fold{f}/features.parquet').set_index('application_id');all_replacement=feature.loc[frame.application_id];query_replacement=feature.loc[q.application_id]
  for t in [1,30]:frame[f're_late_{t}_posterior']=all_replacement[f'cluster_late_{t}'].to_numpy(float);q[f're_late_{t}_posterior']=query_replacement[f'cluster_late_{t}'].to_numpy(float)
  dest=R/f'models/fold{f}';assert not dest.exists();dest.mkdir(parents=True);start=time.time()
  head=feature_head.fit(q.iloc[:6000].assign(target=yc),raw[:6000],oldhead['columns'],.003,.5);assert len(head['coefficients'])==len(oldhead['coefficients'])
  train=frame.iloc[ix['train']].copy();train['target']=labels.loc[train.application_id].to_numpy(np.int64);newsp=spline.fit(train,oldsp['columns']);assert newsp['resolved_parameters']==oldsp['resolved_parameters']and newsp['model'].coef_.shape==oldsp['model'].coef_.shape
  joblib.dump(head,dest/'head.joblib',compress=3);joblib.dump(newsp,dest/'spline.joblib',compress=3)
  def predict(h,s):return .75*b.corrected_probability(h,q.iloc[6000:],raw[6000:])[0]+.25*b.predict_spline(s,frame.iloc[ix['validation']])
  pr=predict(head,newsp);replay=predict(joblib.load(dest/'head.joblib'),joblib.load(dest/'spline.joblib'));assert np.array_equal(pr,replay)
  np.savez_compressed(dest/'validation.npz',rows=ix['validation'],application_id=q.application_id.iloc[6000:].to_numpy(str),prediction=pr)
  b.dump(dict(fold=f,head_rows=6000,spline_rows=56000,head_features=18,spline_features=77,default_labels_opened_for_validation=False,architecture_and_parameters_unchanged=True,source_transformer_hash=b.sha(R/f'fold{f}/transformer.json'),model_replay_exact=True,seconds=time.time()-start),dest/'boundary.json');b.dump(dict(status='complete',files={p.name:b.sha(p)for p in dest.iterdir()if p.is_file()}),dest/'complete.json');emit('fold_complete',fold=f)
 b.dump(dict(status='complete',folds=5,new_models=10,outer_labels_opened=False),R/'TRAINING_COMPLETE.json');report()
def report():
 assert read(R/'TRAINING_COMPLETE.json')['status']=='complete';ref=np.load(CONTROL/'OOF.npz');y=ref['target'].astype(np.int64);fold=ref['fold'];pr=np.full(70000,np.nan);seen=np.zeros(70000,int)
 for f in range(5):
  d=R/f'models/fold{f}'
  for n,h in read(d/'complete.json')['files'].items():assert b.sha(d/n)==h
  z=np.load(d/'validation.npz');rows=z['rows'];assert set(rows)==set(np.flatnonzero(fold==f));pr[rows]=z['prediction'];seen[rows]+=1
 assert np.isfinite(pr).all()and np.all(seen==1);pred=dict(baseline=ref['baseline'],maturity=ref['both'],cluster=pr);mm={k:metric(y,p)for k,p in pred.items()};fm={k:[metric(y[fold==f],p[fold==f])for f in range(5)]for k,p in pred.items()}
 out=dict(status='point_metrics_complete',metrics=mm,fold_metrics=fm,goal_point_reached=mm['cluster']['KS']>=.54,confirmation_opened=False,historical_label_reuse=True,submission_generated=False)
 np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred);b.dump(out,R/'POINT_RESULT.json');emit('point_metrics_complete',metrics=mm)
 groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]];rng=np.random.default_rng(20261119);boot={k:[]for k in ['baseline','maturity']}
 for _ in range(1000):
  check();ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);m={k:b.scores(y[ix],p[ix])for k,p in pred.items()}
  for k in boot:boot[k].append([m['cluster']['KS']-m[k]['KS'],m['cluster']['AUC']-m[k]['AUC']])
 out.update(status='complete',intervals={k:np.quantile(v,[.05/8,1-.05/8],axis=0).tolist()for k,v in boot.items()},uncertainty='1000paired fold/class bootstrap,2contrasts x2metrics,Bonferroni95percent. Reused development labels, conditional only; no adaptive-selection or retraining correction.')
 b.dump(out,R/'result.json');np.savez_compressed(R/'bootstrap.npz',**boot);emit('complete',goal_point_reached=out['goal_point_reached'])
if __name__=='__main__':
 try:main()
 except BaseException:b.dump(dict(status='failed',error=traceback.format_exc()),R/'FAILURE.json');raise
