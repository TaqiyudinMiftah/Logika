"""Freeze inner roles before any new inner model or score is observed."""
from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[1];A=R.parent;P=A/'ranking-pipeline-20261020/input'
p=json.loads((R/'plan.json').read_text());d=R/'input';assert not d.exists();(d/'data').mkdir(parents=True)
(d/'data/train_features.parquet').symlink_to(P/'data/train_features.parquet');(d/'BASELINE_LOCK.json').write_bytes((P/'BASELINE_LOCK.json').read_bytes());data=pd.read_parquet(P/'data/train_features.parquet',columns=['application_id','target']);assert len(data)==70000 and data.application_id.is_unique
jobs={};outer=np.zeros(70000,int)
for f in range(5):
 old=np.load(P/f'artifacts/fold{f}/indices.npz');train=np.sort(np.r_[old['context'],old['correction']]);valid=old['validation'];assert len(train)==56000 and len(valid)==14000 and set(train).isdisjoint(valid);outer[valid]+=1
 seen=np.zeros(70000,int)
 for g,(ti,qi)in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261022+f).split(train,data.target.iloc[train])):
  ci,vi=train[ti],train[qi];j=f*6+g;dest=d/f'artifacts/fold{j}';dest.mkdir(parents=True);np.savez_compressed(dest/'indices.npz',context=ci,validation=vi)
  assert len(ci)==44800 and len(vi)==11200 and set(ci).isdisjoint(vi)and set(ci).isdisjoint(valid);seen[vi]+=1
  jobs[str(j)]=dict(outer_fold=f,inner_fold=g,kind='inner',context_rows=44800,query_rows=11200,indices_sha256=hashlib.sha256((dest/'indices.npz').read_bytes()).hexdigest())
 assert np.all(seen[train]==1)and np.all(seen[valid]==0)
 j=f*6+5;dest=d/f'artifacts/fold{j}';dest.mkdir(parents=True);np.savez_compressed(dest/'indices.npz',context=train,validation=valid);jobs[str(j)]=dict(outer_fold=f,kind='final',context_rows=56000,query_rows=14000,indices_sha256=hashlib.sha256((dest/'indices.npz').read_bytes()).hexdigest())
assert np.all(outer==1);p['jobs']=jobs;(R/'plan.json').write_text(json.dumps(p,indent=2));(R/'ROLE_GATE.json').write_text(json.dumps(dict(status='passed',jobs=30,outer_folds=5,inner_folds=5,inner_queries_cover_each_outertrain_once=True,outer_labels_used_in_split_creation=False,outer_rows=70000,inner_seed='20261022+outerfold'),indent=2))
