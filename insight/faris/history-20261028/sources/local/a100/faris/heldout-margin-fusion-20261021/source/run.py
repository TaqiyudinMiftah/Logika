"""One preregistered heldout fusion. Training has no outer-label input."""
from pathlib import Path
import json,time,argparse,hashlib
import numpy as np,pandas as pd,joblib
import baseline as b
import fusion
R=Path(__file__).resolve().parents[1];ARCHIVE=R.parent
MAIN=ARCHIVE/'kumo-coverage-placement-20261021';STAGED=ARCHIVE/'kumo-temperature-adaptation-20261021/full_data'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):b.dump(x,p)
def source(f,kind,ix,local_manifest):
 d=R/f'imports/{kind}/fold{f}';c=read(d/'complete.json');receipt=read(d/'source_receipt.json')
 assert c['status']=='complete'and c['fold']==f and c['full_replay_max_error']<2e-6
 assert receipt['input_manifest']==local_manifest
 assert receipt['checkpoint_sha256']=={'classification':'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0','regression':'0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'}[kind]
 assert receipt['retained_model_path']
 assert b.sha(d/'raw.npz')==c['files']['raw.npz']
 z=np.load(d/'raw.npz');assert np.array_equal(z['rows'],np.r_[ix['correction'],ix['validation']])
 assert len(z['prediction'])==20000 and np.isfinite(z['prediction']).all()
 return z['prediction']
def train():
 plan=read(R/'plan.json');assert plan['training_admitted']is True
 assert time.time()<plan['original_deadline']-10800
 # All backbone recipes must be complete before fitting this single meta recipe.
 for kind in ['classification','regression']:
  for f in range(5):assert read(R/f'imports/{kind}/fold{f}/complete.json')['status']=='complete'
 for f in range(5):
  assert time.time()<plan['original_deadline']and not(R/'STOP').exists()
  out=R/f'folds/fold{f}';assert not out.exists(),'Preserve earlier attempt';out.mkdir(parents=True)
  data=STAGED/f'fold{f}';manifest=read(data/'manifest.json')
  for n,h in manifest['files'].items():assert b.sha(data/n)==h
  ix=dict(np.load(data/'indices.npz'));ci,hi,vi=[ix[n]for n in ['context','correction','validation']]
  assert len(ci)==50000 and len(hi)==6000 and len(vi)==14000
  assert len(set(np.r_[ci,hi,vi]))==70000 and set(np.r_[ci,hi,vi])==set(range(70000))
  q=pd.read_parquet(data/'query.parquet');assert 'target'not in q and q.application_id.is_unique
  labels=np.load(data/'correction_labels.npy');assert len(labels)==6000
  d=MAIN/f'conditions/fold{f}/control90';c=read(d/'complete.json')
  for name in ['raw_predictions.npz','correction.joblib']:assert b.sha(d/name)==c['small_files'][name]
  # The main campaign already independently verified its full fitted contexts.
  assert read(MAIN/'CACHE_AUDIT.json')['status']=='passed'
  raw=np.load(d/'raw_predictions.npz');assert np.array_equal(raw['rows'],np.r_[hi,vi])
  pk=raw['prediction'];base=joblib.load(d/'correction.joblib')
  pc=source(f,'classification',ix,manifest);pr=source(f,'regression',ix,manifest)
  bundle=fusion.fit(base,q.iloc[:6000],pk[:6000],pc[:6000],pr[:6000],labels)
  expected_hash=hashlib.sha256('\n'.join(q.application_id.iloc[:6000].astype(str)).encode()).hexdigest()
  assert bundle['training_ids_sha256']==expected_hash
  p=fusion.predict(bundle,q.iloc[6000:],pk[6000:],pc[6000:],pr[6000:])
  sp=np.load(data/'spline_validation.npz');assert np.array_equal(sp['rows'],vi);p=.75*p+.25*sp['prediction']
  assert np.isfinite(p).all()and np.all((p>=0)&(p<=1))
  joblib.dump(bundle,out/'model.joblib',compress=3);replay=joblib.load(out/'model.joblib')
  pp=.75*fusion.predict(replay,q.iloc[6000:],pk[6000:],pc[6000:],pr[6000:])+.25*sp['prediction']
  assert np.max(abs(p-pp))<1e-14
  np.savez_compressed(out/'validation.npz',rows=vi,application_id=q.application_id.iloc[6000:].to_numpy(str),prediction=p)
  write(out/'boundary.json',dict(context_rows=50000,head_rows=6000,outer_rows=14000,
   meta_prediction_sources_exclude_all_head_labels=True,spline_prediction_not_used_in_meta_fit=True,
   training_ids_sha256=expected_hash,model_sources={k:read(R/f'imports/{k}/fold{f}/source_receipt.json')for k in ['classification','regression']},
   new_contrast_coefficients=bundle['coefficients'][-2:].tolist(),spec=bundle['spec']))
  write(out/'complete.json',dict(status='complete',fold=f,files={p.name:b.sha(p)for p in out.iterdir()if p.is_file()}))
 write(R/'TRAINING_COMPLETE.json',dict(status='complete',folds=5,outer_label_file_opened=False,correction_fits=5))
def report():
 assert read(R/'TRAINING_COMPLETE.json')['status']=='complete'
 from scipy.stats import ks_2samp,rankdata
 from sklearn.metrics import log_loss
 z=np.load(MAIN/'oof.npz');y=z['target'];fold=z['fold'];p=np.full(70000,np.nan);seen=np.zeros(70000,int)
 for f in range(5):
  d=R/f'folds/fold{f}';c=read(d/'complete.json')
  for n,h in c['files'].items():assert b.sha(d/n)==h
  v=np.load(d/'validation.npz');assert set(v['rows'])==set(np.flatnonzero(fold==f));p[v['rows']]=v['prediction'];seen[v['rows']]+=1
 assert np.all(seen==1)and np.isfinite(p).all();pred={'control90':z['control90'],'fusion':p};metrics={}
 for k,pp in pred.items():
  m=b.scores(y,pp);assert abs(m['KS']-ks_2samp(pp[y==0],pp[y==1]).statistic)<1e-12
  auc=(rankdata(pp)[y==1].sum()-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(y==0));assert abs(auc-m['AUC'])<1e-12
  m['logloss']=float(log_loss(y,pp));metrics[k]=m
 np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred)
 write(R/'result.json',dict(status='complete_point_metrics',metrics=metrics,fold_metrics={k:[b.scores(y[fold==f],pp[fold==f])for f in range(5)]for k,pp in pred.items()},goal_point_reached=metrics['fusion']['KS']>=.54,uncertainty='Paired bootstrap and independent replay required before promotion.',historical_label_reuse=True,submission_generated=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['train','report']);args=a.parse_args();train()if args.action=='train'else report()
