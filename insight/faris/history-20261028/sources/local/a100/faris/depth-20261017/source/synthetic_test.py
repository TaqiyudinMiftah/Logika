import json,tempfile,types
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
import adapter
from metaexp.common import metrics
from scipy.stats import ks_2samp
class FakeBank:
 def __init__(self,*a):
  self.dev=np.arange(180);self.y=np.arange(180)%2;self.folds=np.arange(180)%3
  self.categorical=pd.DataFrame({'employment_type':['e'+str(i) for i in self.dev]})
  self.manifest={'info':{}}
 def columns(self,r):return ['a','b'],['employment_type']
 def cohort(self,fit,rows,names,c):
  rows=np.asarray(rows);return np.column_stack([np.sin(rows),np.where(rows%7==0,np.nan,np.cos(rows))]).astype(np.float32),names
class Ctx:
 def __init__(self,r):
  self.directory=r;self.inputs={'bank':str(r/'bank')};self.split=r/'split.json';self.params={'view':'control','weight':1,'strength':.05};self.family='linear_depth';self.seed=2026;self.spec={'resources':{'cpus':1}};self.models=[]
  self.split.write_text(json.dumps({'development_rows':list(range(180)),'folds':(np.arange(180)%3).tolist()}))
 def check_stop(self):pass
 def path(self,n):p=self.directory/n;p.parent.mkdir(exist_ok=True,parents=True);return p
 def save_json(self,n,v):self.path(n).write_text(json.dumps(v))
 def model(self,n):self.models.append(n)
with tempfile.TemporaryDirectory() as d:
 ctx=Ctx(Path(d))
 with patch('adapter.Bank',FakeBank):
  result=adapter.run_trial(ctx);score=adapter.evaluate(ctx,result);assert len(ctx.models)==3 and np.isfinite(score['score'])
  z=np.load(ctx.path('validation.npz'));p=z['prediction'].copy();p[0]=np.nan;np.savez(ctx.path('validation.npz'),rows=z['rows'],fold=z['fold'],prediction=p)
  try:adapter.evaluate(ctx,result)
  except AssertionError:pass
  else:raise AssertionError('Incomplete OOF accepted')
y=np.array([0,1,0,1,1,0]);p=np.array([.5,.5,.1,.9,.9,.5]);assert abs(metrics(y,p)['ks']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
# Every categorical ID unique: inner target encoding must not recover its own label.
rep=json.loads(Path(__file__).with_name('views.json').read_text())['control'];rep['config']['encoding']='target'
from metaexp.pipeline import Preprocessor
b=FakeBank();pre=Preprocessor(b,rep,'lgb');x=pre.fit_transform(np.arange(120),b.y[:120]);fill=pre.fill.copy();pre.transform(np.arange(120,180));assert np.array_equal(pre.fill,fill) and pre.audit['target_encoding_inner_folds']==3
assert np.all((x[:,-1]>.4)&(x[:,-1]<.6))
print('PASS: full adapter 3 folds, all model retention, evaluator rejects incomplete OOF, ties independent KS, train-only fill, inner target crossfit')
