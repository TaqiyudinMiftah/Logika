from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,numpy as np,pandas as pd,joblib
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b,feature_head
R=Path(__file__).resolve().parents[1];A=R.parent;MAIN=A/'kumo-coverage-placement-20261021';DATA=A/'kumo-temperature-adaptation-20261021/full_data'
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'plan.json')['original_deadline']and not(R/'STOP').exists()
def main():
 p=read(R/'plan.json');assert p['feature_audit_admitted']and not p['training_admitted'];check()
 for n,h in p['source_sha256'].items():assert b.sha(R/'source'/n)==h
 assert read(R/'RESULT.json')['target_labels_opened']is False
 raw=A/'exploration-12h-20261017/raw';ids=pd.concat([pd.read_csv(raw/(s+'_applications.csv'),usecols=['application_id','customer_id'])for s in ['train','test']],ignore_index=True).set_index('application_id');assert ids.index.is_unique
 features={k:pd.read_parquet(R/n).set_index('customer_id')for k,n in [('true','row_local_features.parquet'),('null','null_row_local_features.parquet')]};extras=p['features']
 assert set(features['true'].index)==set(features['null'].index)and all(set(extras)<=set(x)for x in features.values())
 prepared=[];gate=[]
 # No outer labels or global OOF file is opened during input qualification or fitting.
 for f in range(5):
  d=DATA/f'fold{f}';m=read(d/'manifest.json')
  for n,h in m['files'].items():assert b.sha(d/n)==h
  ix=dict(np.load(d/'indices.npz'));q=pd.read_parquet(d/'query.parquet');labels=np.load(d/'correction_labels.npy').astype(np.int64);c=pd.read_parquet(d/'context.parquet',columns=['application_id'])
  assert len(q)==20000 and len(labels)==6000 and'target'not in q and set(q.application_id).isdisjoint(c.application_id)
  source=MAIN/f'conditions/fold{f}/control90';done=read(source/'complete.json')
  for n in ['raw_predictions.npz','correction.joblib']:assert b.sha(source/n)==done['small_files'][n]
  rr=np.load(source/'raw_predictions.npz');assert np.array_equal(rr['rows'],np.r_[ix['correction'],ix['validation']]);old=joblib.load(source/'correction.joblib');assert len(old['columns'])==18
  x,o=b.design(old,q.iloc[:6000],rr['prediction'][:6000]);v,g=b.objective(old['coefficients'],x,o,labels,np.ones(6000),.003);assert abs(v-old['objective'])<1e-12 and max(abs(g))<1e-4
  prepared.append((f,q,labels,ix,rr['prediction'],old));gate.append(dict(fold=f,context=50000,head=6000,outer=14000,original_head_objective_reproduced=True))
 b.dump(dict(status='passed_for_one_dynamic_head_plus_null',checks=gate,outer_labels_opened=False,new_features_shared_statistics=False,feature_sha256={k:b.sha(R/n)for k,n in [('true','row_local_features.parquet'),('null','null_row_local_features.parquet')]}),R/'SCIENTIFIC_GATE.json')
 p['training_admitted']=True;b.dump(p,R/'plan.json')
 for f,q,y,ix,rawp,old in prepared:
  sp=np.load(DATA/f'fold{f}/spline_validation.npz');assert np.array_equal(sp['rows'],ix['validation'])
  for v in ['true','null']:
   check();dest=R/f'folds/{v}/fold{f}';assert not dest.exists();dest.mkdir(parents=True)
   frame=q.copy();customer=ids.customer_id.reindex(frame.application_id);values=features[v][extras].reindex(customer.to_numpy())
   for name in extras:frame[name]=values[name].to_numpy(float)
   assert not{'application_id','customer_id','target','split'}&set(extras)
   head=feature_head.fit(frame.iloc[:6000].assign(target=y),rawp[:6000],old['columns']+extras,.003,.5);joblib.dump(head,dest/'model.joblib',compress=3)
   pr=.75*b.corrected_probability(head,frame.iloc[6000:],rawp[6000:])[0]+.25*sp['prediction'];rp=.75*b.corrected_probability(joblib.load(dest/'model.joblib'),frame.iloc[6000:],rawp[6000:])[0]+.25*sp['prediction'];assert np.max(abs(pr-rp))<1e-14
   np.savez_compressed(dest/'validation.npz',rows=ix['validation'],prediction=pr,application_id=frame.application_id.iloc[6000:].to_numpy(str))
   b.dump(dict(fold=f,variant=v,head_rows=6000,outer_rows=14000,columns=head['columns'],outer_labels_opened=False,gradient_max_abs=head['gradient_max_abs'],parameters=len(head['coefficients'])),dest/'boundary.json');b.dump(dict(status='complete',fold=f,variant=v,files={z.name:b.sha(z)for z in dest.iterdir()if z.is_file()}),dest/'complete.json')
 b.dump(dict(status='complete',heads=10,outer_labels_opened=False),R/'TRAINING_COMPLETE.json')
 report()
def report():
 assert read(R/'TRAINING_COMPLETE.json')['status']=='complete';z=np.load(MAIN/'oof.npz');y=z['target'].astype(np.int64);fold=z['fold'];pred={'control90':z['control90']}
 for v in ['true','null']:
  pr=np.full(70000,np.nan);seen=np.zeros(70000,int)
  for f in range(5):
   d=R/f'folds/{v}/fold{f}';c=read(d/'complete.json')
   for n,h in c['files'].items():assert b.sha(d/n)==h
   a=np.load(d/'validation.npz');assert set(a['rows'])==set(np.flatnonzero(fold==f));pr[a['rows']]=a['prediction'];seen[a['rows']]+=1
  assert np.all(seen==1)and np.isfinite(pr).all();pred[v]=pr
 def metric(y,pr):
  m=b.scores(y,pr);assert abs(m['KS']-ks_2samp(pr[y==0],pr[y==1]).statistic)<1e-12;n1=int(y.sum());n0=len(y)-n1;auc=float((rankdata(pr)[y==1].sum()-n1*(n1+1)/2)/(n1*n0));assert abs(m['AUC']-auc)<1e-12;m['logloss']=float(log_loss(y,pr));return m
 out=dict(status='point_metrics_complete',metrics={k:metric(y,pr)for k,pr in pred.items()},fold_metrics={k:[metric(y[fold==f],pr[fold==f])for f in range(5)]for k,pr in pred.items()},goal_point_reached=metric(y,pred['true'])['KS']>=.54,historical_label_reuse=True,submission_generated=False)
 np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred);b.dump(out,R/'POINT_RESULT.json')
 rng=np.random.default_rng(20261115);groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in[0,1]];boot={k:[]for k in ['control90','null']}
 for _ in range(2000):
  check();ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);m={k:b.scores(y[ix],pr[ix])for k,pr in pred.items()}
  for k in boot:boot[k].append([m['true']['KS']-m[k]['KS'],m['true']['AUC']-m[k]['AUC']])
 out.update(status='complete',conditional_intervals={k:np.quantile(v,[.00625,.99375],axis=0).tolist()for k,v in boot.items()},interval_scope='2000paired fold/class bootstrap, two contrasts/two metrics Bonferroni. Conditional on predictions, does not correct historical search or model-retraining uncertainty.');b.dump(out,R/'result.json')
if __name__=='__main__':main()
