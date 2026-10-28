"""Finite, explicit, deterministic search catalogue, not a Cartesian claim."""
import copy,json
from pathlib import Path
import numpy as np
from utils import ident,read,write

MODELS=['lgb','cat_native','xgb','elastic','spline','ebm','extra','rff','tabm','resmlp','ft','tabr','gru','deepsets','kumo','tabicl','act_logit','act_cloglog','woe_scorecard','cred_scorecard']

def make(root):
 root=Path(root);champ=read(root/'evidence/champion.json');base=copy.deepcopy(next(iter(champ['components']['components'].values()))['recipe']['representation'])
 base['config'].update(select='none',feature_count='all',reducer='none',spline=False,interactions=False,encoding='onehot',indicator=False,transform='identity',window='all',clip=0,missing='median')
 base['family']='source';base['id']='v3';base['config']['_extra_family']=None
 views={'v3':base}
 for key in champ['components']['ensemble']['parents']:
  r=copy.deepcopy(champ['components']['components'][key]['recipe']['representation']);r['config']['_extra_family']=None;views[key]=r
 rng=np.random.default_rng(20261017)
 for family in ['repayment','bureau','account','shopping','cross_source','representation']:
  made=set();j=0
  while j<32:
   r=copy.deepcopy(base);c=r['config'];c['_extra_family']=family;c['window']=[30,90,180,365,'all'][j%5];c['half_life']=[30,90,180][j%3];c['strength']=[0,6,24,48][j%4]
   c['select']=['split','true_gain','variance','correlation'][j%4];c['feature_count']=[64,128,256,'all'][j%4]
   c['indicator']=bool((j//4)%2);c['clip']=[0,.005,.01][j%3]
   c['transform']=['identity','signed_log','rank'][j%3];c['encoding']=['onehot','target','frequency'][j%3]
   if family=='representation':
    c['reducer']=['pca','svd','group_pca','cluster','autoencoder'][j%5];c['dimensions']=[16,32,64][j%3];c['append']=bool(j%2)
   else:c['append']=True
   # Eight single-factor/feature-block anchors precede higher-order combinations.
   if j<8:
    saved=copy.deepcopy(base['config']);saved.update(_extra_family=family,window=[30,90,180,365,'all','all','all','all'][j])
    if j==5:saved.update(select='true_gain',feature_count=128)
    if j==6:saved.update(select='split',feature_count=128)
    if j==7:saved.update(indicator=True)
    if family=='representation':saved.update(reducer=['pca','svd','group_pca','cluster','autoencoder','pca','svd','cluster'][j],dimensions=[16,32,64][j%3],append=j%2==0)
    c.clear();c.update(saved)
   sig=ident(c)
   if sig in made:raise ValueError('Duplicate representation')
   made.add(sig);key=f'{family}_{j:02d}';r.update(id=key,hypothesis='H_'+family,semantic_signature=sig);views[key]=r;j+=1
 assert len(views)==198
 configs={};anchors=[]
 # Champion controls preserve exact settings through dedicated champion model code.
 for k in champ['components']['ensemble']['parents']:
  cfg={'model':'champion','view':k,'weight':6,'anchor':True};cid=ident(cfg)[:20];configs[cid]=cfg;anchors.append(cid)
 for model in MODELS:
  # At least v3 and every feature family get a baseline comparison.
  anchor_views=['v3']+[f'{f}_00' for f in ['repayment','bureau','account','shopping','cross_source','representation']]
  if model in ['lgb','elastic']:anchor_views += [f'{family}_{j:02d}' for family in ['repayment','bureau','account','shopping','cross_source'] for j in [5,6]]
  for view in anchor_views:
   cfg={'model':model,'view':view,'weight':1 if model in ['kumo','tabicl'] else 6,'trees':500,'rate':.04,'strength':40,'leaves':31,'depth':4,'minimum':200,'columns':1.,'width':128,'blocks':2,'dropout':.1,'lr':.0003,'events':32,'context':16384,'estimators':4,'updates':0,'encoding':'ple' if model=='tabm' else 'plain','anchor':True}
   if model in ['elastic','spline','rff','act_logit','act_cloglog']:cfg.update(strength=.05,l1=0)
   if model in ['woe_scorecard','cred_scorecard']:cfg.update(strength=24,bins=5,max_features=64,C=.05,weight=1)
   if model in ['act_logit','act_cloglog']:cfg.update(max_features=128,weight=1)
   cid=ident(cfg)[:20];configs[cid]=cfg;anchors.append(cid)
 keys=list(views)
 while len(configs)<4096:
  model=MODELS[int(rng.integers(len(MODELS)))];view=keys[int(rng.integers(len(keys)))];cfg={'model':model,'view':view,'weight':int(rng.choice([1,3,6])),'anchor':False}
  if model in ['lgb','cat_native','xgb','extra']:
   cfg.update(trees=int(rng.choice([250,500,1000])),rate=float(rng.choice([.02,.04,.08])),strength=float(rng.choice([5,40,100])),leaves=int(rng.choice([15,31,63])),depth=int(rng.choice([3,5,7])),minimum=int(rng.choice([50,200,400])),columns=float(rng.choice([.7,1.])))
  elif model in ['elastic','spline','rff','ebm']:cfg.update(strength=float(rng.choice([.005,.05,.5,5.])),l1=float(rng.choice([0,.25,.75])),dimensions=int(rng.choice([128,256,512])))
  elif model in ['act_logit','act_cloglog']:cfg.update(strength=float(rng.choice([.005,.05,.5,5.])),max_features=int(rng.choice([64,128,256])))
  elif model in ['woe_scorecard','cred_scorecard']:cfg.update(strength=int(rng.choice([6,24,48])),bins=int(rng.choice([3,5,10])),max_features=int(rng.choice([32,64,128])),C=float(rng.choice([.005,.05,.5])))
  elif model in ['kumo','tabicl']:cfg.update(weight=1,context=int(rng.choice([8192,16384,32768,999999])),estimators=int(rng.choice([1,4,8])),updates=int(rng.choice([0,0,0,32,96])) if model=='kumo' else 0,lr=float(rng.choice([1e-6,1e-5])))
  else:cfg.update(width=int(rng.choice([128,256])),blocks=int(rng.choice([2,4])),dropout=float(rng.choice([0,.1,.2])),lr=float(rng.choice([.0001,.0003,.001])),events=int(rng.choice([32,128])),encoding=str(rng.choice(['plain','ple'])) if model=='tabm' else 'plain')
  cid=ident(cfg)[:20];configs.setdefault(cid,cfg)
 write(root/'source/views.json',views);write(root/'source/configurations.json',configs);write(root/'source/anchors.json',anchors)
 write(root/'evidence/catalogue.json',{'representations':198,'new_representations':192,'configurations':len(configs),'families':MODELS,'anchors':anchors,'reference_pool':'explicit finite sampled catalogue','fixed_ensembles_cap':4096,'nested_cap':12})
 return configs
