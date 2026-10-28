"""Standalone frozen Kumo90 + offset correction + spline77 baseline.

Default execution replays actual saved checkpoints against cached frozen-Kumo
scores. It performs no new training and must reproduce the locked CSV bytes.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve,roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent.parent


def sha(path):
 # Completed campaign files are large. Do not retain their read-once pages
 # in the campaign cgroup while repeatedly auditing saved contexts.
 import os
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  bounded=Path(path).resolve().is_relative_to(ROOT.resolve())
  offset=0
  for b in iter(lambda:f.read(8*1024*1024),b''):
   h.update(b)
   if bounded and hasattr(os,'posix_fadvise'):
    os.posix_fadvise(f.fileno(),offset,len(b),os.POSIX_FADV_DONTNEED)
   offset+=len(b)
 return h.hexdigest()


def dump(value,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def fingerprint(frame,columns):return hashlib.sha256(pd.util.hash_pandas_object(frame[columns],index=False).values.tobytes()).hexdigest()
def ids_hash(frame,indices):return hashlib.sha256('\n'.join(frame.application_id.iloc[indices]).encode()).hexdigest()


def scores(y,p):
 f,t,_=roc_curve(y,p,drop_intermediate=False)
 return dict(KS=float(np.max(abs(t-f))),signed_KS=float(np.max(t-f)),AUC=float(roc_auc_score(y,p)),rows=len(y))


def prepare(frame,columns,categorical):
 x=frame[columns].copy()
 for c in categorical:x[c]=x[c].fillna('MISSING').astype(str)
 for c in x.columns.difference(categorical):
  v=x[c].replace([np.inf,-np.inf],np.nan);x[c]=np.sign(v)*np.log1p(abs(v))
 return x


def design(bundle,frame,p):
 with threadpool_limits(limits=4):basis=bundle['encoder'].transform(prepare(frame,bundle['columns'],bundle['categorical']))
 offset=logit(np.clip(np.asarray(p,float),1e-6,1-1e-6))
 x=np.column_stack([np.ones(len(frame)),(offset-bundle['offset_mean'])/bundle['offset_sd'],basis]);assert np.isfinite(x).all()
 return x,offset


def corrected_probability(bundle,frame,p):
 x,offset=design(bundle,frame,p);delta=x@bundle['coefficients']
 return expit(offset+bundle['spec']['correction_strength']*delta),delta


def objective(beta,x,offset,y,weight,penalty,prior_shift=0.):
 eta=offset+prior_shift+x@beta;normalized=weight/weight.sum()
 loss=float(normalized@(np.logaddexp(0,eta)-y*eta)+.5*penalty*(beta[1:]@beta[1:]))
 gradient=x.T@(normalized*(expit(eta)-y));gradient[1:]+=penalty*beta[1:]
 return loss,gradient


def refit_correction(spec,frame,p):
 columns=spec['correction_columns'];cats=frame[columns].select_dtypes(include=['object','string','category']).columns.tolist();nums=[c for c in columns if c not in cats]
 encoder=ColumnTransformer([
  ('smooth',Pipeline([('fill',SimpleImputer(strategy='median')),('basis',SplineTransformer(n_knots=3,degree=2,knots='quantile',extrapolation='constant')),('scale',StandardScaler())]),nums),
  ('missing',MissingIndicator(features='all',sparse=False,error_on_new=False),nums),
  ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
 with threadpool_limits(limits=4):basis=encoder.fit_transform(prepare(frame,columns,cats))
 offset=logit(np.clip(np.asarray(p,float),1e-6,1-1e-6));mean=float(offset.mean());sd=max(float(offset.std()),1e-8)
 x=np.column_stack([np.ones(len(frame)),(offset-mean)/sd,basis]);y=frame.target.to_numpy(int);weight=np.ones(len(y))
 assert spec['balance']=='unweighted' and spec['penalty']==.003 and spec['correction_strength']==.5
 with threadpool_limits(limits=4):
  result=minimize(objective,np.zeros(x.shape[1]),args=(x,offset,y,weight,.003,0.),method='L-BFGS-B',jac=True,options=dict(maxiter=300,ftol=1e-11,gtol=1e-7))
 value,gradient=objective(result.x,x,offset,y,weight,.003,0.);assert result.success and np.max(abs(gradient))<1e-4
 return dict(encoder=encoder,columns=columns,categorical=cats,offset_mean=mean,offset_sd=sd,coefficients=result.x,spec=spec,positive_weight=1.,objective=value,gradient_max_abs=float(np.max(abs(gradient))),iterations=int(result.nit))


def predict_spline(bundle,frame):
 with threadpool_limits(limits=4):
  return bundle['model'].predict_proba(bundle['encoder'].transform(prepare(frame,bundle['columns'],bundle['categorical'])))[:,1]


def replay(refit=False):
 lock=json.loads((ROOT/'BASELINE_LOCK.json').read_text());manifest=json.loads((ROOT/'reports/IMPORT_MANIFEST.json').read_text())
 for item in manifest['artifacts']:assert sha(ROOT/item['destination'])==item['sha256'],item['destination']
 train=pd.read_parquet(ROOT/'data/train_features.parquet');test=pd.read_parquet(ROOT/'data/test_features.parquet');sample=pd.read_csv(ROOT/'data/sample_submission.csv')
 assert len(train)==70000 and len(test)==len(sample)==30000 and train.application_id.is_unique and test.application_id.is_unique
 assert train.customer_id.is_unique and set(train.customer_id).isdisjoint(test.customer_id)
 definition=json.loads((ROOT/'provenance/definition.json').read_text())
 assert fingerprint(train,['application_id','target']+definition['columns'])==definition['training_sha256']
 assert fingerprint(test,['application_id']+definition['columns'])==definition['test_sha256']
 y=train.target.to_numpy(int);oof=np.full(len(train),np.nan);rebuilt=np.full(len(train),np.nan);fold_ids=np.full(len(train),-1);pt=np.zeros(len(test));checks=[]
 for fold in range(5):
  root=ROOT/'artifacts'/f'fold{fold}';index=np.load(root/'indices.npz');ti,vi,ci,hi=[index[k] for k in ['train','validation','context','correction']]
  meta=json.loads((root/'metadata.json').read_text());assert len(ci)==50000 and len(hi)==6000 and len(vi)==14000
  assert not np.intersect1d(ci,hi).size and not np.intersect1d(ti,vi).size and np.array_equal(np.sort(np.r_[ci,hi]),np.sort(ti))
  for key,indices in [('training',ti),('validation',vi),('context',ci),('correction_training',hi)]:assert ids_hash(train,indices)==meta[key+'_ids_sha256']
  raw={kind:pd.read_parquet(root/f'kumo_{kind}.parquet') for kind in ['holdout','oof','test']}
  for kind,query in [('holdout',train.iloc[hi]),('oof',train.iloc[vi]),('test',test)]:assert raw[kind].application_id.tolist()==query.application_id.tolist()
  bundle=joblib.load(root/'correction.joblib')
  if refit:bundle=refit_correction(definition,train.iloc[hi],raw['holdout'].prediction)
  spline=joblib.load(root/'spline.joblib');errors=[];corrector_errors=[]
  for kind,query in [('oof',train.iloc[vi]),('test',test)]:
   corrected,delta=corrected_probability(bundle,query,raw[kind].prediction)
   sp=predict_spline(spline,query);cached_sp=pd.read_parquet(root/f'spline_{kind}.parquet')
   assert cached_sp.application_id.tolist()==query.application_id.tolist();assert np.max(abs(sp-cached_sp.prediction))<2e-14
   prediction=.75*corrected+.25*sp;expected=pd.read_parquet(root/f'final_{kind}.parquet')
   assert expected.application_id.tolist()==query.application_id.tolist()
   error=float(np.max(abs(prediction-expected.prediction)));corrector_errors.append(float(np.max(abs(delta-expected.correction_logit))));errors.append(error)
   if not refit:assert error<2e-14 and corrector_errors[-1]<2e-14
   if kind=='oof':oof[vi]=prediction;rebuilt[vi]=.75*raw[kind].prediction.to_numpy(float)+.25*sp;fold_ids[vi]=fold
   else:pt+=pd.Series(prediction,index=test.application_id).reindex(sample.application_id).to_numpy()/5
  x,offset=design(bundle,train.iloc[hi],raw['holdout'].prediction);loss,gradient=objective(bundle['coefficients'],x,offset,y[hi],np.ones(len(hi)),.003)
  assert abs(loss-bundle['objective'])<1e-12 and np.max(abs(gradient))<1e-4
  checks.append(dict(fold=fold,metrics=scores(y[vi],oof[vi]),three_way_split_disjoint=True,correction_parameters=len(bundle['coefficients']),
   prediction_max_abs_error=max(errors),correction_max_abs_error=max(corrector_errors),objective=loss,gradient_max_abs=float(np.max(abs(gradient)))))
  print(json.dumps(checks[-1]),flush=True)
 assert np.isfinite(oof).all() and np.isfinite(pt).all() and np.all((pt>=0)&(pt<=1))
 metrics=scores(y,oof);payload=pd.DataFrame(dict(application_id=sample.application_id,default_probability=pt)).to_csv(index=False).encode()
 digest=hashlib.sha256(payload).hexdigest();original=ROOT/'submissions'/lock['submission_file']
 if not refit:assert digest==lock['submission_sha256'] and payload==original.read_bytes()
 output=ROOT/'reproduced'/('submission_refit_correction.csv' if refit else lock['submission_file']);output.parent.mkdir(exist_ok=True);output.write_bytes(payload)
 pd.DataFrame(dict(application_id=train.application_id,target=y,fold=fold_ids,prediction=oof)).to_csv(ROOT/'reports/OOF.csv' if not refit else ROOT/'reproduced/OOF_refit.csv',index=False)
 if not refit:pd.DataFrame(dict(application_id=train.application_id,target=y,fold=fold_ids,prediction=rebuilt)).to_csv(ROOT/'reports/OOF_uncorrected_context_control.csv',index=False)
 result=dict(status='PASS',mode='refit_correction' if refit else 'checkpoint_replay',metrics=metrics,fold_checks=checks,
  submission_sha256=digest,byte_identical_to_locked_submission=digest==lock['submission_sha256'],new_Kumo_training=False,
  all_files_verified=True,all_ids_disjoint=True,trusted_point_in_time=False,public_score=lock['public_score'] if digest==lock['submission_sha256'] else None)
 dump(result,ROOT/'reports'/('REFIT_AUDIT.json' if refit else 'REPLAY_AUDIT.json'));print(json.dumps(result,indent=2),flush=True)
 return result


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--refit-correction',action='store_true');args=p.parse_args();replay(args.refit_correction)
