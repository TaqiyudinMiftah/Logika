from pathlib import Path
import shutil,itertools,json,hashlib,py_compile
P=Path('a100/faris/overnight-refit-20261030');F=P/'book-feature-cann';R=P/'woe-feature-cann';R.mkdir(exist_ok=True)
for n in ['source','ops','evidence']:shutil.copytree(F/n,R/n,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for n in ['INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','VALIDATION_CONTRACT.json','CONFIG.json','AGENTS.md','PARENT_CLOCK_REFERENCE.json']:(R/n).write_bytes((F/n).read_bytes())
(R/'source/woe_features.py').write_text('''"""Cross-fitted smoothed bad-risk enrichment, a signed WoE/credibility adaptation.

Only inner training labels enter maps. OOF assignments are label-independent and
held fixed; own-row labels cannot enter that row's encoded training features.
"""
import numpy as np,pandas as pd
from scipy import sparse
from scipy.special import logit
from sklearn.model_selection import KFold
from cpu_offset import FrozenInput,_hash,_ids
class WoeInput(FrozenInput):
 def __init__(self,mode,bins,strength):self.mode=mode;self.bins=int(bins);self.strength=float(strength);assert mode in {'append','woe_only'}
 def fit_supervised(self,frame,rows,y):
  rows=np.asarray(rows,np.int64);y=np.asarray(y,np.int64);assert len(rows)==len(y) and set(np.unique(y))=={0,1}
  super().fit(frame,rows);self.edges={};self.categories={};self.sizes={}
  for c in self.numeric:
   a=pd.to_numeric(frame[c].iloc[rows],errors='coerce').to_numpy(dtype=float,na_value=np.nan);a=a[np.isfinite(a)]
   self.edges[c]=np.unique(np.quantile(a,np.linspace(0,1,self.bins+1)[1:-1]))if len(a)else np.array([])
   self.sizes[c]=len(self.edges[c])+2
  for c in self.categorical:
   a=frame[c].iloc[rows].astype('string').fillna('__MISSING__');levels=a.value_counts().head(63).index.tolist();self.categories[c]={str(v):i for i,v in enumerate(levels)};self.sizes[c]=len(levels)+1
  codes=self._codes(frame,rows);self.maps=self._fit_maps(codes,y);self.oof=np.zeros(codes.shape);fold=np.empty(len(rows),dtype=np.int64);self.crossfit_row_hashes=[]
  for k,(tr,va) in enumerate(KFold(3,shuffle=True,random_state=20261030).split(rows)):
   maps=self._fit_maps(codes[tr],y[tr]);self.oof[va]=self._lookup(codes[va],maps);fold[va]=k
   self.crossfit_row_hashes.append({'train_rows_sha256':_ids(rows[tr]),'held_rows_sha256':_ids(rows[va])})
  self.positions=np.full(len(frame),-1,dtype=np.int64);self.positions[rows]=np.arange(len(rows));self.fold_hash=_hash(fold);self.target_hash=_hash(y);return self
 def _codes(self,frame,rows):
  out=[]
  for c in self.columns:
   if c in self.edges:
    a=pd.to_numeric(frame[c].iloc[rows],errors='coerce').to_numpy(dtype=float,na_value=np.nan);v=np.searchsorted(self.edges[c],a,side='right');v[~np.isfinite(a)]=self.sizes[c]-1
   else:
    a=frame[c].iloc[rows].astype('string').fillna('__MISSING__');v=a.map(self.categories[c]).fillna(self.sizes[c]-1).to_numpy(dtype=np.int64)
   out.append(v)
  return np.column_stack(out).astype(np.int64)
 def _fit_maps(self,codes,y):
  prior=float(np.clip(np.mean(y),1e-6,1-1e-6));out=[]
  for j,c in enumerate(self.columns):
   n=np.bincount(codes[:,j],minlength=self.sizes[c]);k=np.bincount(codes[:,j],weights=y,minlength=self.sizes[c]);p=(k+self.strength*prior)/(n+self.strength)
   out.append(np.clip(logit(np.clip(p,1e-8,1-1e-8))-logit(prior),-8,8))
  return out
 def _lookup(self,codes,maps):return np.column_stack([m[codes[:,j]]for j,m in enumerate(maps)])
 def glm(self,frame,rows):
  rows=np.asarray(rows,np.int64);z=self._lookup(self._codes(frame,rows),self.maps);pos=self.positions[rows];known=pos>=0;z[known]=self.oof[pos[known]];assert np.isfinite(z).all()
  if self.mode=='append':return sparse.hstack([super().glm(frame,rows),sparse.csr_matrix(z)],format='csr')
  missing=~np.isfinite(self._numeric(frame,rows));return sparse.hstack([sparse.csr_matrix(z),sparse.csr_matrix(missing.astype(float))],format='csr')
 def recipe(self):
  v=super().recipe();v.update(feature_engineering_mode='crossfit_bad_risk_enrichment_'+self.mode,bins=self.bins,beta_binomial_prior_strength=self.strength,folds=3,fold_seed=20261030,label_independent_folds=True,fold_sha256=self.fold_hash,crossfit_rows=self.crossfit_row_hashes,permitted_inner_target_sha256=self.target_hash,full_inner_map_sha256=[_hash(x)for x in self.maps],OOF_feature_sha256=_hash(self.oof),numeric_edges_sha256={c:_hash(v)for c,v in self.edges.items()},own_row_label_excluded=True,retains_inner_encoder_at_full_context_refit=True,no_global_supervised_rescaling=True,fitted_weights_saved=False);return v
''')
s=(F/'source/cpu_models.py').read_text().replace('from book_features import BookInput as FrozenInput','from woe_features import WoeInput')
s=s.replace("encoder=FrozenInput(mode=spec['book_transform']).fit(frame,ctx[inner])", "encoder=WoeInput(mode=spec['woe_mode'],bins=spec['woe_bins'],strength=spec['woe_strength']).fit_supervised(frame,ctx[inner],y[inner])")
s=s.replace("model_family='BookFeature_Bernoulli_CANN'", "model_family='Crossfit_WoE_Bernoulli_CANN'")
(R/'source/cpu_models.py').write_text(s)
c=json.loads((F/'CONFIG.json').read_text());designs={}
for view,offset,width,bins,strength,mode in itertools.product(['actuarial99','global99'],['intercept','glm'],[64,128],[8,16],[24,100],['append','woe_only']):
 name=f'woe_{view}_{offset}_w{width}_b{bins}_s{strength}_{mode}'
 designs[name]=dict(admission_group=name,view=view,offset_mode=offset,width=width,glm_C=.1,residual_penalty=.1,learning_rate=.0003,book_transform='crossfit_woe',woe_mode=mode,woe_bins=bins,woe_strength=strength,max_seconds=900,parameters={},family='Crossfit_WoE_Bernoulli_CANN',resource='cpu',hypothesis='Cross-fitted univariate credit-risk enrichment can regularize sparse feature effects before nonlinear residual learning. Compare appended vs encoded-only inputs, smoothness and bin counts, with archived rawCANN controls; prior standalone WOE scorecards are distinct.',sources=['user:Credit-Scoring--Theory,-Methods,-and-Practice.pdf#PDFpage197','prior:exploration-12h-20261017/README.md#standalone-WOE-scorecards'])
c.update(campaign='woe-feature-cann',designs=designs,primary=next(iter(designs)),pilot_tasks=[f'{d}__s2026__f0'for d in designs],priority_designs=list(designs),expected_fits=len(designs)*10,notes='64designs/640plannedfits. Label-independent3fold crossfit42500inner; no supervised feature stats on7500stopper/6000meta/14000outer. Prior standalone WOE scorecard family acknowledged; this tests composition with neural CANN.')
for k,v in list(c.items()):
 if isinstance(v,list)and any(isinstance(x,str)and x.startswith('book_')for x in v):c[k]=list(designs)
(R/'CONFIG.json').write_text(json.dumps(c,indent=2)+'\n')
plan=json.loads((F/'EXECUTION_PLAN.json').read_text());owners=list(plan['owners']);cycle=sum(([o]*plan['owners'][o]['slots']for o in owners),[])
for owner in owners:
 names=[d for j,d in enumerate(designs)if cycle[j%len(cycle)]==owner];plan['owners'][owner].update(designs=names,pilot_tasks=[f'{d}__s2026__f0'for d in names],planned_fits=len(names)*10)
(R/'EXECUTION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');(R/'EXTRA_DESIGNS.jsonl').write_text('')
contract=json.loads((F/'VALIDATION_CONTRACT.json').read_text());contract.update(protocol_id='logika-v9-crossfit-woe-cann64',feature_policy='Training-only supervised riskmap: quantile/category bins from42500inner covariates. Beta-binomial prevalence-centred bad-logodds, three label-independent KFold maps excluding own label. Stop/meta/outer features usefull42500inner map; fresh50kmodelretainsencoder. Same original bank; new derived representation allowed by current human scope.')
(R/'VALIDATION_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
s=(P/'localglm-cann/ops/localglm_coordinator.py').read_text().replace('localglm0630','woe0630').replace("P/'book-feature-cann/private/owner_metadata'","P/'localglm-cann/private/owner_metadata'").replace('Bernoulli_LocalGLMnet','Crossfit_WoE_Bernoulli_CANN')
(R/'ops/woe_coordinator.py').write_text(s)
(R/'README.md').write_text('''# Cross-fitted WoE and credibility features with CANN
64designs/640plannedfits, not completion. Twoexisting99-columnviews × GLM/intercept × widths64/128 ×8/16quantilebins ×24/100priorstrength × append/encoded-only. Fixedlr.0003/residualpenalty.1. Weight-of-evidence motivated by suppliedCreditScoring PDFpage197; beta-binomial-smoothed bad-risk logodds minuspriorlogit is a signed/adapted WoE, not an exact reproduction of unsmoothed good/bad formula. PriorOctober17 standaloneWOE/credibility scorecards are acknowledged; nonlinear CANN interaction is the new question. All bin/category definitions use42500inner covariates;3fold shuffledKFold(20261030) independentoflabels; eachinner training row uses riskmap from other2folds, excluding ownlabel. Stop/meta/outer use fullinner map, no heldoutlabels. No global label-dependent normalization. Retains inner encoder for fresh50kmodel. Synthetic own-label-mutation, heldout-label-mutation, missing/constant/unseen checks gate admission. All70k adaptively reused, no independent gain claim. Per-owner20slots waits priorLocalGLMownerFINAL; unchanged06:30WIB absolute ceiling,05:30admission/06:00model cutoff. No fittedweights/KVcaches. Higher audited localwinner may trigger fresh70krefit only within original reserve. CurrentCSV is never downgraded and V25CSV is never a fallback.
''')
for p in (R/'source').glob('*.py'):py_compile.compile(str(p),doraise=True)
(R/'SOURCE_SEAL.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((R/'source').rglob('*.py'))},indent=2)+'\n')
print(json.dumps({'designs':len(designs),'planned_fits':len(designs)*10}))
