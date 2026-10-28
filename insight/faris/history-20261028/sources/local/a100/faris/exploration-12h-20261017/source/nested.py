"""Genuine nested meta-learning; global OOF predictions never fit meta weights."""
import copy
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression,Ridge
from scipy.optimize import nnls
from features import ExploreBank
from utils import read,ks

def run(ctx):
 from adapter import fold_fit
 bank=ExploreBank(Path(ctx.inputs['bank']).parent,ctx.inputs['extras']);configs=read(Path(__file__).with_name('configurations.json'));views=read(Path(__file__).with_name('views.json'));ids=ctx.params['components'];assert 2<=len(ids)<=3 and len(set(ids))==len(ids) and all(i in configs for i in ids)
 assert all(configs[i]['model'] in ['champion','lgb','elastic','spline','rff'] for i in ids)
 mode=int(ctx.params['config'].split('_')[-1])%3
 folds=np.asarray(read(ctx.split).get('folds',bank.folds));confirm=ctx.spec['protocol']=='confirm';outrows=np.load(bank.root/'confirmation_rows.npy') if confirm else bank.dev
 output=np.full(len(outrows),np.nan)
 outer=[(np.arange(len(bank.y)),np.arange(len(outrows)))] if confirm else [(np.flatnonzero(folds!=f),np.flatnonzero(folds==f)) for f in range(3)]
 for f,(ti,vi) in enumerate(outer):
  trainrows=bank.dev[ti];queryrows=outrows[vi];y=bank.y[ti];inner=np.full((len(ti),len(ids)),np.nan);query=np.empty((len(vi),len(ids)));seed=ctx.seed+f
  split=list(StratifiedKFold(3,shuffle=True,random_state=seed+700).split(ti,y))
  for j,key in enumerate(ids):
   p=copy.deepcopy(configs[key]);rep=copy.deepcopy(views[p['view']]);rep['config']['weight']=p['weight']
   for g,(a,b) in enumerate(split):inner[b,j]=fold_fit(ctx,bank,p,rep,trainrows[a],y[a],trainrows[b],f'outer_{f}/component_{key}/inner_{g}',seed+g)
   query[:,j]=fold_fit(ctx,bank,p,rep,trainrows,y,queryrows,f'outer_{f}/component_{key}/refit',seed)
  assert np.isfinite(inner).all()
  if mode==0:
   meta=LogisticRegression(C=.1,max_iter=1000);meta.fit(inner,y);pred=meta.predict_proba(query)[:,1]
  elif mode==1:
   weights=nnls(inner,y)[0];weights=weights/weights.sum() if weights.sum()>0 else np.repeat(1/len(ids),len(ids));meta={'weights':weights};pred=query@weights
  else:
   meta=Ridge(alpha=100);meta.fit(inner,y);pred=np.clip(meta.predict(query),1e-8,1-1e-8)
  name=f'outer_{f}/meta.joblib';joblib.dump(meta,ctx.path(name));ctx.model(name);np.savez_compressed(ctx.path(f'outer_{f}/inner_oof.npz'),rows=trainrows,prediction=inner);output[vi]=pred
  ctx.save_json(f'outer_{f}/lineage.json',{'components':ids,'inner_folds':3,'outer_training_rows':len(ti),'outer_validation_rows':len(vi),'meta_mode':mode,'no_global_oof_training':True})
 values={'rows':outrows,'prediction':np.clip(output,1e-8,1-1e-8)}
 if not confirm:values['fold']=folds
 np.savez_compressed(ctx.path('validation.npz'),**values);return {'validation':['validation.npz'],'nested':True,'folds':len(outer)}
