"""Last preparation gate: versions, bank/source hashes and champion protection."""
import importlib.metadata,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'));from utils import read,write,sha,ident
host=sys.argv[1]
for base,manifest in [(ROOT/'source','source_manifest.json'),(ROOT,'management_manifest.json')]:
 for name,checksum in read(ROOT/manifest).items():assert sha(base/name)==checksum,('Frozen code changed',name)
config=read(ROOT/'deployment.json');bank=Path(config['hosts'][host]['bank']);m=read(bank/'manifest.json')
assert m['bank_id']==read(ROOT/'evidence/champion.json')['components']['bank_id']
for name,checksum in m['files'].items():assert sha(bank/name)==checksum,('Bank changed',name)
packages=['numpy','pandas','scipy','scikit-learn','pyarrow','lightgbm','catboost','xgboost','joblib','optuna','psutil','interpret-core']
if host in ['pc2','pc3']:packages+=['torch','tabm','rtdl-num-embeddings']
versions={n:importlib.metadata.version(n) for n in packages};(ROOT/'source/requirements.lock').write_text(''.join(n+'=='+v+'\n' for n,v in versions.items()))
if host=='pc2':
 for name,checksum in m['raw_sha256'].items():assert sha(ROOT/'raw'/(name+'.csv'))==checksum
 prior=Path('/home/faris/faris/submission-research-20261017');old=read(prior/'artifacts/complete.json');assert old['submission_sha256']==read(ROOT/'evidence/champion_identity.json')['submission_sha256']
 protected=ROOT/'archive/champion';protected.mkdir(parents=True,exist_ok=True)
 import shutil
 for name in ['artifacts','frozen_recipe.json','frozen_recipe.sha256','source']:
  p=prior/name
  if p.is_dir():shutil.copytree(p,protected/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
  else:shutil.copy2(p,protected/name)
 for folder,checksum in old['models'].items():assert sha(protected/'artifacts'/folder/'pipeline.joblib')==checksum
 write(ROOT/'evidence/champion_protection.json',{'submission_sha256':old['submission_sha256'],'models':old['models'],'archive':str(protected),'immutable_champion_preserved':True})
write(ROOT/'evidence'/('freeze-'+host+'.json'),{'passed':True,'time':time.time(),'versions':versions,'bank_id':m['bank_id'],'bank_files_verified':len(m['files']),'source_manifest_sha256':sha(ROOT/'source_manifest.json'),'management_manifest_sha256':sha(ROOT/'management_manifest.json')})
print(json.dumps(read(ROOT/'evidence'/('freeze-'+host+'.json'))))
