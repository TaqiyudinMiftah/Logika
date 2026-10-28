"""Full-CV recipes, retained transformers and models, independent persisted evaluation."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
os.environ['MPLBACKEND']='Agg'
import copy,gc,json,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from features import ExploreBank,Processor
from metaexp.pipeline import fit_predict,predict
from metaexp.common import json_safe
from utils import read,write,sha,ks,ident
NEURAL=['tabm','resmlp','ft','tabr','gru','deepsets']

def load(ctx):
 bank=ExploreBank(Path(ctx.inputs['bank']).parent,ctx.inputs['extras']);cat=read(Path(__file__).with_name('configurations.json'));cid=ctx.params['config'];p=copy.deepcopy(cat[cid]);rep=copy.deepcopy(read(Path(__file__).with_name('views.json'))[p['view']]);rep['config']['weight']=p['weight'];return bank,p,rep

def model(p,seed,cpus,gpu,amd=False):
 k=p['model'];w=p['weight'];trees=p.get('trees',500)
 if k in ['act_logit','act_cloglog']:
  from actuarial import BinomialGLM
  return BinomialGLM(link='logit' if k=='act_logit' else 'cloglog',strength=p.get('strength',.05),weight=w,max_features=p.get('max_features',128),seed=seed)
 if k in ['woe_scorecard','cred_scorecard']:
  from actuarial import CredibilityScorecard
  return CredibilityScorecard(bins=p.get('bins',5),strength=p.get('strength',24),weight=w,max_features=p.get('max_features',64),credibility=k=='cred_scorecard',C=p.get('C',.05),seed=seed)
 if k=='lgb':
  from lightgbm import LGBMClassifier
  return LGBMClassifier(n_estimators=trees,learning_rate=p.get('rate',.04),num_leaves=p.get('leaves',31),min_child_samples=p.get('minimum',200),reg_lambda=p.get('strength',40),colsample_bytree=p.get('columns',1),scale_pos_weight=w,n_jobs=cpus,verbosity=-1,random_state=seed,**({'device_type':'gpu','gpu_platform_id':0,'gpu_device_id':0,'max_bin':63} if amd else {}))
 if k=='cat_native':
  from catboost import CatBoostClassifier
  return CatBoostClassifier(iterations=trees,depth=p.get('depth',4),learning_rate=p.get('rate',.04),l2_leaf_reg=p.get('strength',40),class_weights=[1,w],thread_count=cpus,random_seed=seed,verbose=False,allow_writing_files=False,**({'task_type':'GPU','devices':'0','gpu_ram_part':.85} if gpu else {}))
 if k=='xgb':
  from xgboost import XGBClassifier
  return XGBClassifier(n_estimators=trees,max_depth=p.get('depth',4),learning_rate=p.get('rate',.04),min_child_weight=p.get('minimum',200),reg_lambda=p.get('strength',40),colsample_bytree=p.get('columns',1),scale_pos_weight=w,n_jobs=cpus,random_state=seed,tree_method='hist',device='cuda' if gpu else 'cpu')
 if k=='extra':
  from sklearn.ensemble import ExtraTreesClassifier
  return ExtraTreesClassifier(n_estimators=min(trees,500),max_depth=12,min_samples_leaf=50,class_weight={0:1,1:w},n_jobs=cpus,random_state=seed)
 if k=='ebm':
  from interpret.glassbox import ExplainableBoostingClassifier
  return ExplainableBoostingClassifier(max_rounds=1000,outer_bags=4,interactions=10,n_jobs=cpus,random_state=seed)
 from sklearn.linear_model import LogisticRegression
 return LogisticRegression(C=p.get('strength',.05),class_weight={0:1,1:w},max_iter=1500,tol=.0002,solver='saga' if k=='elastic' else 'lbfgs',penalty='elasticnet' if k=='elastic' else 'l2',l1_ratio=p.get('l1',.25) if k=='elastic' else None,random_state=seed)

def save_pre(ctx,folder,pre):
 pre.detach();name=folder+'/preprocessing.joblib';joblib.dump(pre,ctx.path(name),compress=1);ctx.model(name);ctx.save_json(folder+'/preprocessing.json',pre.audit)

def fold_fit(ctx,bank,p,rep,rows,y,query,folder,seed):
 ctx.check_stop();kind=p['model'];cpus=ctx.spec['resources']['cpus'];gpu=bool(ctx.spec['resources'].get('gpus'));amd=ctx.family.endswith('_opencl')
 if kind=='champion':
  original=read(Path(__file__).with_name('champion.json'));recipe=original['components']['components'][p['view']]['recipe'];values,pre,m,w,cdf,warnings=fit_predict(bank,recipe,rows,y,query,'cpu',seed)
  pre.bank=None;name=folder+'/pipeline.joblib';joblib.dump({'preprocessing':pre,'model':m,'weight':w},ctx.path(name),compress=1);ctx.model(name);ctx.save_json(folder+'/audit.json',{'preprocessing':pre.audit,'parameters':json_safe(m.get_params()),'warnings':warnings});return values
 if kind in NEURAL:
  from sklearn.model_selection import train_test_split
  import neural
  ti,vi=train_test_split(np.arange(len(rows)),test_size=.125,stratify=y,random_state=seed+4123)
  prep=Processor(bank,rep,kind,seed);x=prep.fit_transform(rows[ti],y[ti]);v=prep.transform(rows[vi]);events=qevents=None
  def ev(a,fitrows):
   e=np.array(np.load(Path(ctx.inputs['extras'])/'events.npy',mmap_mode='r')[a,:p.get('events',32)],dtype=np.float32);l=np.minimum(np.load(Path(ctx.inputs['extras'])/'lengths.npy')[a],p.get('events',32));return e,l
  if kind in ['gru','deepsets']:events=ev(rows[ti],rows[ti]);qevents=ev(rows[vi],rows[ti])
  elif kind=='tabr':
   from sklearn.neighbors import NearestNeighbors
   search=NearestNeighbors(n_neighbors=17,n_jobs=cpus).fit(x);ii=search.kneighbors(x,return_distance=False);ii=np.stack([a[a!=j][:16] for j,a in enumerate(ii)]);vv=search.kneighbors(v,n_neighbors=16,return_distance=False);context=np.column_stack([y[ti][ii].mean(1),y[ti][ii].std(1)]);x=np.column_stack([x,context]);v=np.column_stack([v,y[ti][vv].mean(1),y[ti][vv].std(1)])
   name=folder+'/inner/retrieval.joblib';joblib.dump({'index':search,'training_targets':y[ti],'self_exclusion':True},ctx.path(name),compress=1);ctx.model(name)
  save_pre(ctx,folder+'/inner',prep);_,epochs=neural.train(ctx,kind,x,y[ti],v,folder+'/inner',p,events,qevents,valid=y[vi],seed=seed)
  prep=Processor(bank,rep,kind,seed);x=prep.fit_transform(rows,y);q=prep.transform(query);events=qevents=None
  if kind in ['gru','deepsets']:events=ev(rows,rows);qevents=ev(query,rows)
  elif kind=='tabr':
   from sklearn.neighbors import NearestNeighbors
   search=NearestNeighbors(n_neighbors=17,n_jobs=cpus).fit(x);ids=search.kneighbors(x,return_distance=False)
   # Explicit self removal, robust to tied distances.
   ids=np.stack([a[a!=j][:16] for j,a in enumerate(ids)]);qi=search.kneighbors(q,n_neighbors=16,return_distance=False);x=np.column_stack([x,y[ids].mean(1),y[ids].std(1)]);q=np.column_stack([q,y[qi].mean(1),y[qi].std(1)])
   name=folder+'/retrieval.joblib';joblib.dump({'index':search,'training_targets':y,'training_rows':rows},ctx.path(name),compress=1);ctx.model(name)
  save_pre(ctx,folder,prep);values,_=neural.train(ctx,kind,x,y,q,folder,p,events,qevents,epochs=epochs,seed=seed);return values
 if kind in ['kumo','tabicl']:
  from foundation import fit_foundation
  return fit_foundation(ctx,bank,p,rep,rows,y,query,folder,seed)
 pre=Processor(bank,rep,kind,seed);x=pre.fit_transform(rows,y);q=pre.transform(query)
 if kind=='cat_native':
  # Native categorical columns alongside the engineered numeric view.
  cats=bank.categorical.columns.tolist();x=pd.DataFrame(x);q=pd.DataFrame(q);x.columns=x.columns.astype(str);q.columns=q.columns.astype(str)
  for c in cats:x['cat_'+c]=bank.categorical[c].iloc[rows].to_numpy(dtype=str);q['cat_'+c]=bank.categorical[c].iloc[query].to_numpy(dtype=str)
 elif kind in ['spline','rff']:
  from sklearn.preprocessing import SplineTransformer
  from sklearn.kernel_approximation import RBFSampler
  tr=SplineTransformer(n_knots=5,degree=3) if kind=='spline' else RBFSampler(n_components=p.get('dimensions',256),gamma=1/max(x.shape[1],1),random_state=seed)
  if kind=='spline':x=x[:,:64];q=q[:,:64]
  x=tr.fit_transform(x);q=tr.transform(q);name=folder+'/extra_transform.joblib';joblib.dump(tr,ctx.path(name));ctx.model(name)
 m=model(p,seed,cpus,gpu,amd);fitkw={'cat_features':['cat_'+c for c in cats]} if kind=='cat_native' else {}
 if kind=='ebm':fitkw['sample_weight']=np.where(y==1,p['weight'],1.)
 m.fit(x,y,**fitkw)
 if kind in ['act_logit','act_cloglog','woe_scorecard','cred_scorecard']:
  audit={'outcome':'12-month 90-DPD default probability','model':kind,'training_rows':len(y),'selected_preprocessed_indices':m.columns_.tolist(),'parameters':m.get_params(),'no_severity_or_survival_target':True}
  if hasattr(m,'optimization_'):audit.update(optimizer_success=bool(m.optimization_.success),iterations=int(m.optimization_.nit),coefficients=m.coef_.tolist(),link=m.link)
  if hasattr(m,'maps_'):audit.update(inner_crossfit_folds=3,maps=[{'edges':a.tolist(),'woe':b.tolist(),'posterior_sd':c.tolist(),'credibility_weight':d.tolist()} for a,b,c,d in m.maps_])
  ctx.save_json(folder+'/actuarial.json',audit)
 values=predict(m,q,p['weight']);name=folder+'/model.joblib';joblib.dump(m,ctx.path(name),compress=1);ctx.model(name);ctx.save_json(folder+'/parameters.json',json_safe(m.get_params()));save_pre(ctx,folder,pre);return values

def run_trial(ctx):
 if ctx.family=='nested':
  from nested import run
  return run(ctx)
 bank,p,rep=load(ctx);split=read(ctx.split);folds=np.asarray(split.get('folds',bank.folds));seed=ctx.seed
 if ctx.spec['protocol']=='confirm':
  query=np.load(bank.root/'confirmation_rows.npy');pred=fold_fit(ctx,bank,p,rep,bank.dev,bank.y,query,'confirmation',seed);np.savez_compressed(ctx.path('validation.npz'),rows=query,prediction=pred)
 else:
  assert np.array_equal(bank.dev,split['development_rows']);assert len(folds)==len(bank.y)
  if ctx.protocol.get('temporal'):
   times=bank.metadata.decision_datetime.iloc[bank.dev].to_numpy();order=np.argsort(times,kind='stable');cut=int(.8*len(order));vi=order[cut:];ti=order[:cut];ti=ti[times[ti]<times[vi].min()];pred=fold_fit(ctx,bank,p,rep,bank.dev[ti],bank.y[ti],bank.dev[vi],'temporal',seed);np.savez_compressed(ctx.path('validation.npz'),rows=bank.dev[vi],prediction=pred)
  else:
   pred=np.full(len(bank.y),np.nan)
   for f in range(3):
    ctx.check_stop();ti=np.flatnonzero(folds!=f);vi=np.flatnonzero(folds==f);assert not np.intersect1d(bank.dev[ti],bank.dev[vi]).size
    start=time.time();pred[vi]=fold_fit(ctx,bank,p,rep,bank.dev[ti],bank.y[ti],bank.dev[vi],f'fold_{f}',seed+f)
    np.savez_compressed(ctx.path(f'fold_{f}/validation.npz'),rows=bank.dev[vi],prediction=pred[vi]);ctx.save_json(f'fold_{f}/identity.json',{'fit_rows':ident(bank.dev[ti].tolist()),'query_rows':ident(bank.dev[vi].tolist()),'seed':seed+f,'seconds':time.time()-start,'host':os.uname().nodename,'gpu_visibility':os.environ.get('CUDA_VISIBLE_DEVICES')})
    print(json.dumps({'fold':f,'ks':ks(bank.y[vi],pred[vi]),'seconds':time.time()-start}),flush=True)
   np.savez_compressed(ctx.path('validation.npz'),rows=bank.dev,fold=folds,prediction=pred)
 return {'validation':['validation.npz'],'folds':3 if ctx.spec['protocol']!='confirm' and not ctx.protocol.get('temporal') else 1,'parameters':p,'representation':rep}

def evaluate(ctx,result):
 bank=ExploreBank(Path(ctx.inputs['bank']).parent,ctx.inputs['extras']);z=np.load(ctx.path('validation.npz'));values=z['prediction'];rows=z['rows'];assert len(rows)==len(np.unique(rows)) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
 if ctx.spec['protocol']=='confirm':
  assert np.array_equal(rows,np.load(bank.root/'confirmation_rows.npy'));y=np.load(bank.root/'confirmation_y.npy')
 else:
  ix=pd.Index(bank.dev).get_indexer(rows);assert (ix>=0).all();y=bank.y[ix]
  if not ctx.protocol.get('temporal'):assert np.array_equal(rows,bank.dev) and np.array_equal(z['fold'],read(ctx.split)['folds'])
 from scipy.stats import ks_2samp
 score=ks(y,values);ind=float(ks_2samp(values[y==0],values[y==1]).statistic);assert abs(score-ind)<1e-12
 from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,roc_curve
 from actuarial import reliability
 return {**reliability(y,values),'score':score,'per_fold':[ks(y[z['fold']==f],values[z['fold']==f]) for f in range(3)] if 'fold' in z else [],'directional_ks':float(np.max(roc_curve(y,values)[1]-roc_curve(y,values)[0])),'auc':float(roc_auc_score(y,values)),'average_precision':float(average_precision_score(y,values)),'log_loss':float(log_loss(y,values)),'independent_ks':ind,'caveat':'Historical adaptive reuse; not a public score'}
