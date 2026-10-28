"""Fixed supervised representation; labels from context50000 only."""
from pathlib import Path
import sys,os,json,time,hashlib,argparse
R=Path(__file__).resolve().parents[1];A=R.parent
sys.path.insert(0,str(A/'catboost-convergence-20261021/source'))
from worker import load,sha
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool

def run(f):
 p=json.loads((R/'plan.json').read_text());assert p['selection_admitted']and time.time()<p['original_deadline']-10800
 d=R/f'data/fold{f}';assert not d.exists();d.mkdir(parents=True)
 tr,va,ix,s,m=load(f);assert np.array_equal(ix['train'],np.r_[ix['context'],ix['correction']]);context=tr.iloc[:50000]
 cats=s['categorical'];params=dict(p['selector']);params.update(task_type='GPU',devices='0',thread_count=4,gpu_ram_part=.85,random_seed=2026+f,train_dir=str(d/'logs'))
 model=CatBoostClassifier(**params);model.fit(Pool(context[s['columns']],context.target,cat_features=cats),verbose=100);model.save_model(str(d/'selector.cbm'))
 importance=model.get_feature_importance(type='PredictionValuesChange');assert len(importance)==1032 and np.isfinite(importance).all()
 order=sorted(range(1032),key=lambda i:(-importance[i],s['columns'][i]));numeric=[n for n in s['columns']if n not in cats]
 x=context[numeric].astype(np.float64);x=x.fillna(x.median()).fillna(0);v=x.to_numpy();v-=v.mean(axis=0);norm=np.sqrt(np.sum(v*v,axis=0));good=norm>1e-10;v[:,good]/=norm[good];v[:,~good]=0;pos={n:i for i,n in enumerate(numeric)}
 selected=[];num_selected=[];rejected=[]
 for i in order:
  name=s['columns'][i]
  if name in pos:
   col=pos[name]
   if not good[col]:rejected.append([name,'constant_in_context']);continue
   if num_selected and np.max(np.abs(v[:,num_selected].T@v[:,col]))>=.95:rejected.append([name,'context_absolute_Pearson_at_least0.95']);continue
   num_selected.append(col)
  elif context[name].nunique(dropna=False)<=1:rejected.append([name,'constant_in_context']);continue
  selected.append(name)
  if len(selected)==90:break
 assert len(selected)==90 and not{'target','application_id','customer_id','split'}&set(selected)
 head=json.loads((R/'head_columns.json').read_text());columns=list(dict.fromkeys(selected+head))
 context[['application_id','target']+selected].to_parquet(d/'context.parquet',index=False)
 query=pd.concat([tr.iloc[50000:][['application_id']+columns],va[['application_id']+columns]],ignore_index=True);assert 'target'not in query and set(context.application_id).isdisjoint(query.application_id)
 query.to_parquet(d/'query.parquet',index=False);np.save(d/'correction_labels.npy',tr.target.iloc[50000:].to_numpy(np.int8));np.savez_compressed(d/'indices.npz',**ix)
 (d/'columns.json').write_text(json.dumps(selected,indent=2));(d/'selection.json').write_text(json.dumps(dict(method='CatBoost PredictionValuesChange; descending importance, then greedy context-only Pearson0.95 pruning',fit_rows=50000,reserved6000_used_in_selection=False,outer_labels_available=False,selected=selected,importance={s['columns'][i]:float(importance[i])for i in order},rejected=rejected),indent=2))
 (d/'manifest.json').write_text(json.dumps(dict(status='complete',fold=f,input_manifest=m,selector_model_sha256=sha(d/'selector.cbm'),feature_selection_rows=50000,outer_labels_used=False,files={n:sha(d/n)for n in ['context.parquet','query.parquet','correction_labels.npy','indices.npz','columns.json','selection.json']}),indent=2))
 print(json.dumps(dict(fold=f,status='selection_complete',selected=90)),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--fold',type=int,required=True);run(a.parse_args().fold)
