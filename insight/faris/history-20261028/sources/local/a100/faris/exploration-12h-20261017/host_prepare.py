"""Build/seal per-host finite queues. Dataset transforms only run after clock start."""
import argparse,importlib.metadata,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'));from utils import read,write,sha

def prepare(host):
 c=read(ROOT/'deployment.json');paths=c['hosts'][host];source=ROOT/'source';env={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','pyarrow','lightgbm','catboost','xgboost','joblib','optuna','psutil','interpret-core']}
 if host in ['pc2','pc3']:env.update({n:importlib.metadata.version(n) for n in ['torch','tabm','rtdl-num-embeddings']})
 if host=='node380':
  for n in ['torch','structured-data-models','cloudpickle','tabicl']:
   try:env[n]=importlib.metadata.version(n)
   except importlib.metadata.PackageNotFoundError:pass
 (source/'requirements.lock').write_text(''.join(k+'=='+v+'\n' for k,v in env.items()))
 configs=read(source/'configurations.json');anchors=read(source/'anchors.json');families=[];omitted=[]
 from features import ExploreBank
 from utils import ident
 b=ExploreBank(Path(paths['bank']).parent,ROOT/'extras');views=read(source/'views.json');signatures={}
 for key,rep in views.items():
  names,cats=b.columns(rep);active={k:v for k,v in rep['config'].items() if k in ['missing','clip','indicator','transform','scale','encoding','select','feature_count','reducer','dimensions','append','spline','cohort_weight','strength','category_normalize']}
  if active['select']=='none' and active['feature_count']=='all':active.pop('feature_count')
  if active['reducer']=='none':active.pop('dimensions');active.pop('append')
  if rep['config'].get('_extra_family')!='repayment':active.pop('strength',None)
  if active['select']=='gain':active['select']='split'
  signatures[key]=ident([names,cats,active])
 seen=set();deduplicated=[];kept={}
 for key in anchors+[k for k in sorted(configs) if k not in anchors]:
  p=configs[key];m=p['model'];allowed={'model','weight'}
  if m=='champion':allowed|={'view'}
  elif m in ['lgb','cat_native','xgb','extra']:allowed|={'trees','rate','strength','leaves','depth','minimum','columns'}
  elif m in ['elastic','spline','rff','ebm']:allowed|={'strength','l1','dimensions'}
  elif m in ['act_logit','act_cloglog']:allowed|={'strength','max_features'}
  elif m in ['woe_scorecard','cred_scorecard']:allowed|={'strength','bins','max_features','C'}
  elif m in ['kumo','tabicl']:allowed|={'context','estimators','updates','lr'}
  else:allowed|={'width','blocks','dropout','lr','events','encoding'}
  if m=='lgb':allowed-= {'depth'}
  if m=='cat_native':allowed-= {'leaves','minimum','columns'}
  if m=='xgb':allowed-= {'leaves'}
  if m=='extra':allowed-= {'rate','strength','leaves','depth','minimum','columns'}
  if m not in ['gru','deepsets']:allowed.discard('events')
  if m!='tabm':allowed.discard('encoding')
  if m=='elastic':allowed.discard('dimensions')
  if m in ['spline','rff']:allowed.discard('l1')
  if m=='spline':allowed.discard('dimensions')
  if m=='ebm':allowed-={'strength','l1','dimensions'}
  active={k:v for k,v in p.items() if k in allowed}
  if m=='extra':active['trees']=min(active.get('trees',500),500)
  if m=='ft':active['width']=min(active.get('width',128),128)
  if m=='tabicl' or (m=='kumo' and active.get('updates',0)==0):active.pop('lr',None)
  signature=ident([signatures[p['view']],active])
  if signature in seen:deduplicated.append(key);continue
  seen.add(signature);kept[key]=p
 configs=kept;write(ROOT/'evidence/resolved-deduplication.json',{'registered':4096,'unique_resolved':len(kept),'excluded_equivalent':deduplicated,'intentional_historical_controls':5})

 for model in sorted({p['model'] for p in configs.values()}):
  if host=='node380' and model=='tabicl' and not (ROOT/'weights/tabicl-classifier-v2-20260212.ckpt').exists():omitted.append({'family':'tabicl','reason':'Checkpoint not staged before admission'});continue
  ids=[]
  for key,p in configs.items():
   if p['model']!=model:continue
   cpu=model in ['elastic','spline','ebm','extra','rff','act_logit','act_cloglog','woe_scorecard','cred_scorecard'];gpu=model in ['cat_native','xgb','tabm','resmlp','ft','tabr','gru','deepsets'];i=int(key,16)
   target='node380' if model in ['kumo','tabicl'] else (('pc1' if read(ROOT/'evidence/champion.json')['components']['components'][p['view']]['recipe']['model']=='lgb' else 'pc2') if model=='champion' else (['pc1','pc2','pc3'][i%3] if cpu else ('pc1' if model=='lgb' else ['pc2','pc3'][i%2])))
   if target==host:ids.append(key)
  if not ids:continue
  gpu=model in ['cat_native','xgb','tabm','resmlp','ft','tabr','gru','deepsets','kumo','tabicl']
  ram=9000 if gpu else (5000 if model in ['extra','ebm'] else 2800)
  resources={'cpus':2 if gpu else 1,'ram_mb':ram,'disk_mb':500 if model in ['extra','tabm','ft','gru','deepsets','resmlp','tabr'] else 180,'timeout_seconds':2400 if gpu else 1500}
  if gpu:resources.update(gpus=1,vram_mb=9500 if host!='node380' else 55000)
  families.append({'name':model,'hypothesis':'H_'+model+'; matched champion and feature-block exploration','parameters':{'config':{'type':'categorical','choices':ids}},'resources':resources,'anchors':[{'params':{'config':key}} for key in anchors if key in ids]})
 if host=='pc1':
  lgb=next(f for f in families if f['name']=='lgb');families.remove(lgb)
  import copy
  ids=lgb['parameters']['config']['choices']
  for suffix,index in [('cpu',0),('opencl',1)]:
   f=copy.deepcopy(lgb);f['name']='lgb_'+suffix;f['parameters']['config']['choices']=ids[index::2];f['anchors']=[a for a in f['anchors'] if a['params']['config'] in f['parameters']['config']['choices']]
   if suffix=='opencl':f['resources']['opencl']=1
   families.append(f)
 if host=='pc2':families.append({'name':'nested','hypothesis':'H7 genuine nested ensembles','parameters':{'config':{'type':'categorical','choices':['nested_'+str(i) for i in range(12)]}},'resources':{'cpus':1,'ram_mb':6000,'disk_mb':800,'timeout_seconds':4500},'stage_only':True})
 if host=='node380':
  for f in families:f['resources'].update(cpus=min(4,paths['cpus']),ram_mb=min(40000,int(paths['ram_mb']*.75)),disk_mb=6000,timeout_seconds=4500,vram_mb=55000)
 assert families
 conf={'version':1,'id':'explore12h-'+host,'competition_root':str(ROOT),'adapter':'adapter.py','metric':'tie-aware KS','direction':'maximize','inputs':{'bank':paths['bank'],'extras':str(ROOT/'extras')},'brief':{'description':'12h maximum exploration, full three-fold screening; champion public user report 0.56188','rules':str(ROOT/'evidence/rules.md'),'compute_access':host+' all authorized compatible hardware, preserve other processes','external_data_policy':'No external borrower data','pretrained_policy':'Official permitted frozen weights, offline staged on node380 only'},'evidence':{'research':str(ROOT/'evidence/research.md'),'eda':str(ROOT/'evidence/eda.json'),'hypotheses':str(ROOT/'evidence/hypotheses.md'),'applicability':str(ROOT/'evidence/applicability.md')},'protocols':{'screen':{'split':str(ROOT/'evidence/screen.json')},'reseed':{'split':str(ROOT/'evidence/screen.json')},'repeat':{'split':str(ROOT/'evidence/repeat.json')},'temporal':{'split':str(ROOT/'evidence/screen.json'),'temporal':True},'confirm':{'split':str(ROOT/'evidence/confirmation_split.json')}},'families':families,'budget_seconds':43200,'storage_limit_mb':paths['storage_mb'],'max_trials':sum(len(f['parameters']['config']['choices']) for f in families if not f.get('stage_only')),'max_workers':paths['cpus'],'cpu_limit':paths['cpus'],'ram_limit_mb':paths['ram_mb'],'seeds':[2026],'search_seed':20261017+['pc1','pc2','pc3','node380'].index(host),'reserve_fraction':.25,'breadth_fraction':.40,'coverage_pool_limit':4096,'disk_headroom_mb':2048,'max_retries':1,'family_failure_limit':3,'poll_seconds':1,'external_clock':str(ROOT/'clock.json'),'exploration_fraction':.30,'opencl':host=='pc1','stage_tasks':str(ROOT/'stage_tasks.json')}
 if host=='node380' and any(p.is_file() for p in (ROOT/'weights').glob('*')):conf['inputs']['weights']=str(ROOT/'weights')
 write(ROOT/'evidence/host-exclusions.json',omitted+[{'family':'exact_T abR'.replace(' ',''),'reason':'Local TabR-inspired retrieval baseline is implemented; faithful official training port not tested and excluded'},{'family':'TabPFN-2.5','reason':'Checkpoint access and model terms not resolved before host admission; no unattended acceptance of terms'}])
 write(ROOT/'input.json',conf)
 if not (ROOT/'campaign').exists():subprocess.run([sys.executable,str(ROOT/'toolkit/campaign.py'),'bundle','--config',str(ROOT/'input.json'),'--source',str(source),'--output',str(ROOT/'campaign')],check=True)
 subprocess.run([sys.executable,str(ROOT/'campaign/runtime/campaign.py'),'validate',str(ROOT/'campaign')],check=True)
 write(ROOT/'prepared.json',{'host':host,'families':[f['name'] for f in families],'registered':conf['max_trials'],'input_sha256':sha(ROOT/'input.json'),'env':env})
 print(json.dumps(read(ROOT/'prepared.json')))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('host');args=a.parse_args();prepare(args.host)
