"""Fixed18-field corrector: baseline, maturity control, joint severity replacement."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,traceback,fcntl
import numpy as np
import pandas as pd
import joblib
from scipy.stats import dirichlet_multinomial,ks_2samp,rankdata
from scipy.special import gammaln
from sklearn.metrics import log_loss
import baseline as b,feature_head
from build import dm_loss,projected_gradient
R=Path(__file__).resolve().parents[1];A=R.parent;MAIN=A/'kumo-coverage-placement-20261021';DATA=A/'kumo-temperature-adaptation-20261021/full_data';INPUT=A/'ranking-pipeline-20261020/input'
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'plan.json')['original_deadline']and not(R/'STOP').exists()
def emit(stage,**v):print(json.dumps(dict(stage=stage,time=time.time(),**v)),flush=True)
def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 n=int(y.sum());auc=(rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n));assert abs(auc-m['AUC'])<1e-12
 m['logloss']=float(log_loss(y,p));return m
def main():
 lock=(R/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
 p=read(R/'plan.json');assert p['classifier_training_admitted']and not(R/'TRAINING_COMPLETE.json').exists()
 for n,h in p['source_sha256'].items():assert b.sha(R/'source'/n)==h
 audit=read(R/'BUILD_COMPLETE.json');assert audit['target_labels_opened']is False and audit['threshold_counts_independently_verified']
 assert b.sha(R/'source/build.py')==audit['build_source_sha256']
 assert read(A/'feature-only-20261021/BASELINE_REPRODUCTION.json')['status']=='passed'
 counts=pd.read_parquet(R/'severity_counts.parquet');allc=counts[[f'count{i}'for i in range(4)]].to_numpy(float)
 gates=[]
 for f in range(5):
  params=read(R/f'fold{f}/transformer.json');assert params['source_count_hash']==b.sha(R/'severity_counts.parquet')and params['feature_hash']==b.sha(R/f'fold{f}/features.parquet')
  ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');ctx=ix['context'];train=allc[ctx];train=train[train.sum(axis=1)>0]
  aa=np.asarray(params['dirichlet_alpha']);v,g=dm_loss(np.log(aa),train);assert abs(v-params['objective'])<1e-10 and max(abs(projected_gradient(np.log(aa),g)))<1e-4
  example=train[:128];coefficient=gammaln(example.sum(axis=1)+1)-gammaln(example+1).sum(axis=1)
  independent=-float((dirichlet_multinomial.logpmf(example,aa,example.sum(axis=1))-coefficient).mean())
  assert abs(independent-dm_loss(np.log(aa),example)[0])<1e-10
  orig=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);assert params['fit_ids_sha256']==b.ids_hash(orig,ctx)
  feat=pd.read_parquet(R/f'fold{f}/features.parquet');assert feat.application_id.tolist()==counts.application_id.tolist()
  posterior=(allc+aa)/(allc.sum(axis=1)[:,None]+aa.sum())
  for threshold,bins in [(1,[1,2,3]),(30,[3])]:assert np.allclose(feat[f'joint_late_{threshold}'],posterior[:,bins].sum(axis=1),rtol=0,atol=1e-14)
  gates.append(dict(fold=f,prior_fit_rows=50000,prior_fit_ids_verified=True,prior_objective_independent_SciPy_check=True,posterior_reconstruction=True,outer_labels_opened=False))
 b.dump(dict(status='passed',folds=gates,prior_fit_target_labels_opened=False),R/'SCIENTIFIC_GATE.json')
 for f in range(5):
  check();d=DATA/f'fold{f}';m=read(d/'manifest.json')
  for n,h in m['files'].items():assert b.sha(d/n)==h
  q=pd.read_parquet(d/'query.parquet');y=np.load(d/'correction_labels.npy').astype(np.int64);ix=np.load(d/'indices.npz')
  original_ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz')
  for role in ['context','correction','validation']:assert np.array_equal(ix[role],original_ix[role])
  assert len(y)==6000 and len(q)==20000 and'target'not in q and not set(ix['context'])&set(ix['correction'])and not set(ix['validation'])&set(np.r_[ix['context'],ix['correction']])
  source=MAIN/f'conditions/fold{f}/control90';done=read(source/'complete.json')
  for n in ['raw_predictions.npz','correction.joblib','validation.npz']:assert b.sha(source/n)==done['small_files'][n]
  raw=np.load(source/'raw_predictions.npz');assert np.array_equal(raw['rows'],np.r_[ix['correction'],ix['validation']]);pr=raw['prediction'];old=joblib.load(source/'correction.joblib');assert len(old['columns'])==18
  feat=pd.read_parquet(R/f'fold{f}/features.parquet').set_index('application_id').loc[q.application_id];sp=np.load(d/'spline_validation.npz');assert np.array_equal(sp['rows'],ix['validation'])
  for name in ['baseline','matched_maturity','joint']:
   check();dest=R/f'heads/{name}/fold{f}';assert not dest.exists();dest.mkdir(parents=True);frame=q.copy();start=time.time()
   for threshold in [1,30]:
    if name!='baseline':frame[f're_late_{threshold}_posterior']=feat[f'{"matched_beta"if name=="matched_maturity"else"joint"}_late_{threshold}'].to_numpy(float)
   model=feature_head.fit(frame.iloc[:6000].assign(target=y),pr[:6000],old['columns'],.003,.5);assert len(model['coefficients'])==len(old['coefficients'])
   joblib.dump(model,dest/'model.joblib',compress=3);valid=frame.iloc[6000:]
   pred=.75*b.corrected_probability(model,valid,pr[6000:])[0]+.25*sp['prediction'];replay=.75*b.corrected_probability(joblib.load(dest/'model.joblib'),valid,pr[6000:])[0]+.25*sp['prediction'];assert np.array_equal(pred,replay)
   if name=='baseline':assert np.max(abs(pred-np.load(source/'validation.npz')['prediction']))<2e-6
   np.savez_compressed(dest/'validation.npz',rows=ix['validation'],prediction=pred,application_id=valid.application_id.to_numpy(str))
   b.dump(dict(fold=f,variant=name,fit_rows=6000,features=18,parameters=len(model['coefficients']),same_parameter_count_as_baseline=True,fit_seconds=time.time()-start,query_labels_passed_to_fit=False,source_prior_hash=b.sha(R/f'fold{f}/transformer.json')),dest/'boundary.json')
   b.dump(dict(status='complete',files={z.name:b.sha(z)for z in dest.iterdir()if z.is_file()}),dest/'complete.json');emit('head_complete',fold=f,variant=name)
 b.dump(dict(status='complete',heads=15,outer_labels_opened=False),R/'TRAINING_COMPLETE.json');report()
def report():
 assert read(R/'TRAINING_COMPLETE.json')['status']=='complete';z=np.load(MAIN/'oof.npz');y=z['target'].astype(np.int64);fold=z['fold'];pred={}
 for name in ['baseline','matched_maturity','joint']:
  p=np.full(70000,np.nan);seen=np.zeros(70000,int)
  for f in range(5):
   d=R/f'heads/{name}/fold{f}'
   for n,h in read(d/'complete.json')['files'].items():assert b.sha(d/n)==h
   v=np.load(d/'validation.npz');rows=v['rows'];assert set(rows)==set(np.flatnonzero(fold==f));p[rows]=v['prediction'];seen[rows]+=1
  assert np.isfinite(p).all()and np.all(seen==1);pred[name]=p
 mm={k:metric(y,p)for k,p in pred.items()};fm={k:[metric(y[fold==f],p[fold==f])for f in range(5)]for k,p in pred.items()}
 assert abs(mm['baseline']['KS']-.5302185366909258)<2e-6
 contrasts={name:{k:mm['joint'][k]-mm[name][k]for k in ['KS','AUC','logloss']}for name in ['baseline','matched_maturity']}
 for name in contrasts:contrasts[name]['fold_KS_deltas']=[fm['joint'][f]['KS']-fm[name][f]['KS']for f in range(5)]
 out=dict(status='point_metrics_complete',metrics=mm,fold_metrics=fm,contrasts=contrasts,goal_point_reached=mm['joint']['KS']>=.54,historical_label_reuse=True,confirmation_opened=False,submission_generated=False)
 np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred);b.dump(out,R/'POINT_RESULT.json');emit('point_metrics_complete',metrics=mm)
 groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]];rng=np.random.default_rng(20261117);boot={k:[]for k in contrasts}
 for _ in range(1000):
  check();ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);m={k:b.scores(y[ix],v[ix])for k,v in pred.items()}
  for k in boot:boot[k].append([m['joint']['KS']-m[k]['KS'],m['joint']['AUC']-m[k]['AUC']])
 out.update(status='complete',intervals={k:np.quantile(v,[.05/8,1-.05/8],axis=0).tolist()for k,v in boot.items()},uncertainty='1000 paired fold/class bootstrap. Two contrasts x two metrics, Bonferroni95percent. Conditional on predictions; historical selection/retraining uncertainty excluded.')
 b.dump(out,R/'result.json');np.savez_compressed(R/'bootstrap.npz',**boot);emit('complete',goal_point_reached=out['goal_point_reached'])
if __name__=='__main__':
 try:main()
 except BaseException:b.dump(dict(status='failed',error=traceback.format_exc()),R/'FAILURE.json');raise
