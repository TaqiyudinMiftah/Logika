import json,sys,time,subprocess,importlib.metadata,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
host=sys.argv[1];bank=Path(sys.argv[2]).resolve()
source=ROOT/'source';evidence=ROOT/'evidence';campaign=ROOT/(sys.argv[3] if len(sys.argv)>3 else 'campaign')
sys.path.insert(0,str(source))
from metaexp.features import Bank
b=Bank(bank.parent)
assert len(b.y)==56000 and set(b.folds)=={0,1,2}
assert len(set(b.dev))==len(b.dev)
(evidence/'split.json').write_text(json.dumps({'development_rows':b.dev.tolist(),'folds':b.folds.tolist(),'confirmation':'excluded; historical and adaptive reuse disclosed'}))
# Confirm expected bank identity without ever opening confirmation labels.
selection=json.loads((evidence/'selection.json').read_text())
assert b.manifest['bank_id']==selection['eda']['bank_id']
views=selection['views'];(source/'views.json').write_text(json.dumps(views,indent=2))
(source/'requirements.lock').write_text(''.join(f'{n}=={importlib.metadata.version(n)}\n' for n in ['numpy','pandas','scipy','scikit-learn','pyarrow','lightgbm','catboost','xgboost','joblib','optuna','psutil']))
# EDA uses development covariates only. No target-driven inspection of confirmation.
import numpy as np
x=b.numeric[b.dev[::20]]
(evidence/'eda.json').write_text(json.dumps({'rows':len(b.y),'fold_counts':np.bincount(b.folds).tolist(),'positive_rate':float(np.mean(b.y)),'covariate_sample_rows':len(x),'features':x.shape[1],'missing_rate':float(np.isnan(x).mean()),'bank_id':b.manifest['bank_id'],'feature_lineage':b.manifest['info'],'historical_selection':selection['selection']},default=str))
models={'pc1':['linear','lgb'],'pc2':['lgb','cat','xgb'],'pc3':['linear','cat','xgb'],'node380':['linear','lgb','cat','xgb']}[host]
# Different machines search disjoint depth/rate strata; baseline anchors intentionally repeat.
strata={'pc1':([250,500,900],[.02,.035,.06]),'pc2':([450,800,1200],[.025,.05]),'pc3':([600,1000,1600],[.02,.04]),'node380':([800,1600,2400],[.015,.03,.06])}
trees,rates=strata[host];families=[]
def cat(x):return {'type':'categorical','choices':x}
for model in models:
 params={'view':cat(list(views)),'variant':cat(['original','no_reduction','no_spline','indicator','no_interactions']),'weight':cat([1,3,6])}
 if model=='linear':params.update(strength=cat([.005,.02,.05,.2,1,5]))
 else:
  params.update(trees=cat(trees),rate=cat(rates),strength=cat([5,25,50,100]))
  if model=='lgb':params.update(leaves=cat([15,31,63]),minimum=cat([80,200,400]),columns=cat([.7,1.]))
  elif model=='xgb':params.update(depth=cat([3,5,7]),minimum=cat([20,80,160]),columns=cat([.7,1.]))
  else:params.update(depth=cat([4,6,8]))
 anchor={k:v['choices'][0] for k,v in params.items()};anchor.update(view='control',variant='original',weight=6,strength=.05 if model=='linear' else 25)
 gpu=model in ['cat','xgb']
 resources={'cpus':2 if gpu else 1,'ram_mb':5000 if gpu else 2500,'disk_mb':150 if gpu else 70,'timeout_seconds':1500 if host=='node380' else 900}
 if gpu:resources.update(gpus=1,vram_mb=18000 if host=='node380' else 8500)
 anchors=[{'params':anchor,'ablation':{'group':model+'-view','factor':'view','control':True}}]
 for v in list(views)[1:5]:anchors.append({'params':dict(anchor,view=v),'ablation':{'group':model+'-view','factor':'view','control':False}})
 families.append({'name':model+'_depth','hypothesis':'H1 stronger training on historical promising views; H2 matched removal sensitivity; H3 regularization and weighting','parameters':params,'resources':resources,'anchors':anchors,'exclude':[{'view':v,'variant':mode} for v,r in views.items() for mode,key,disabled in [('no_reduction','reducer','none'),('no_spline','spline',False),('no_interactions','interactions',False),('indicator','indicator',True)] if r['config'][key]==disabled]})
limits={'pc1':(16,50000,6500),'pc2':(12,32000,5000),'pc3':(12,50000,26000),'node380':(16,160000,100000)}
cpu,ram,disk=limits[host]
c={'version':1,'id':'depth-20261017-'+host,'competition_root':str(ROOT),'adapter':'adapter.py','metric':'tie-aware KS','direction':'maximize','inputs':{'bank':str(bank)},'brief':{'description':'Depth search until 06:00 WIB, supplied data only; all models retained','rules':str(evidence/'rules.md'),'compute_access':host+'; authorized all available resources subject to actual capacity','external_data_policy':'none','pretrained_policy':'none'},'evidence':{'research':str(evidence/'research.md'),'eda':str(evidence/'eda.json'),'hypotheses':str(evidence/'hypotheses.md'),'applicability':str(evidence/'applicability.md')},'protocols':{'screen':{'split':str(evidence/'split.json')}},'families':families,'budget_seconds':max(1,1792191600-time.time()-90),'storage_limit_mb':disk,'max_trials':1500,'max_workers':cpu,'cpu_limit':cpu,'ram_limit_mb':ram,'seeds':[2026],'search_seed':{'pc1':3801,'pc2':3802,'pc3':3803,'node380':3804}[host],'reserve_fraction':.05,'breadth_fraction':.3,'coverage_pool_limit':4096,'disk_headroom_mb':2048,'max_retries':0,'family_failure_limit':3,'poll_seconds':1}
(ROOT/'input.json').write_text(json.dumps(c,indent=2))
subprocess.run([sys.executable,str(ROOT/'toolkit/campaign.py'),'bundle','--config',str(ROOT/'input.json'),'--source',str(source),'--output',str(campaign)],check=True)
subprocess.run([sys.executable,str(campaign/'runtime/campaign.py'),'validate',str(campaign)],check=True)
print(json.dumps({'host':host,'prepared':True,'absolute_deadline':1792191600,'budget':c['budget_seconds']}))
