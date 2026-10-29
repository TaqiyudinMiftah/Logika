from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import numpy as np,pandas as pd
from woe_features import WoeInput
rng=np.random.default_rng(20261030);n=180
frame=pd.DataFrame({'tail':np.r_[rng.lognormal(0,2,n-2),np.nan,np.inf],'constant':np.ones(n),'missing':np.full(n,np.nan),'category':np.array(['a','b','c']*(n//3),dtype=object)})
frame.loc[179,'category']='UNSEEN';rows=np.arange(150);y=(np.arange(n)%3==0).astype(int);checks=[]
for mode in ['append','woe_only']:
 a=WoeInput(mode,8,24).fit_supervised(frame,rows,y[rows]);v=a.glm(frame,np.arange(n)).toarray();assert np.isfinite(v).all()
 for index in [0,13,74,149]:
  changed=y.copy();changed[index]=1-changed[index];b=WoeInput(mode,8,24).fit_supervised(frame,rows,changed[rows]);assert np.array_equal(a.glm(frame,np.array([index])).toarray(),b.glm(frame,np.array([index])).toarray()),(mode,index)
 changed=y.copy();changed[150:]=1-changed[150:];b=WoeInput(mode,8,24).fit_supervised(frame,rows,changed[rows]);assert np.array_equal(v,b.glm(frame,np.arange(n)).toarray())
 # Changing heldout covariates cannot affect train binning or train features.
 other=frame.copy();other.loc[150:,'tail']=1e99;c=WoeInput(mode,8,24).fit_supervised(other,rows,y[rows]);assert np.array_equal(a.glm(frame,rows).toarray(),c.glm(other,rows).toarray())
 checks.append(dict(mode=mode,finite=True,own_label_mutation_exclusion=True,heldout_label_exclusion=True,heldout_covariate_exclusion=True,missing_constant_unseen=True,features=v.shape[1]))
(R/'SYNTHETIC_WOE_GATE.json').write_text(json.dumps(dict(status='PASS',checks=checks,scope='Fixed inner membership and label-independent encoder folds; no global label-dependent scaling; full model data gate still required'),indent=2)+'\n');print(json.dumps(checks))
