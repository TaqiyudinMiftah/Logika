"""Host-local start within shared clock; no dependency on the laptop."""
import os,sys,time,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'));from utils import read,write,sha
for base,manifest in [(ROOT/'source','source_manifest.json'),(ROOT,'management_manifest.json')]:
 for name,checksum in read(ROOT/manifest).items():assert sha(base/name)==checksum,('Frozen source changed',name)
host=sys.argv[1];clock=read(ROOT/'clock.json');deadline=clock['deadline']
if time.time()>=deadline:raise SystemExit(0)
if host=='pc2' and not (ROOT/'features-distributed.json').exists():
 import tarfile,shlex
 from prepare_features import build
 cfg=read(ROOT/'deployment.json');bank=Path(cfg['hosts']['pc2']['bank'])
 if not (ROOT/'extras/manifest.json').exists():build(ROOT/'raw',bank,ROOT/'extras')
 import numpy as np
 from sklearn.model_selection import StratifiedKFold
 from features import ExploreBank
 b=ExploreBank(bank.parent,ROOT/'extras');repeat=np.empty(len(b.y),np.int8)
 for f,(_,ix) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261018).split(b.dev,b.y)):repeat[ix]=f
 write(ROOT/'evidence/screen.json',{'development_rows':b.dev.tolist(),'folds':b.folds.tolist()})
 write(ROOT/'evidence/repeat.json',{'development_rows':b.dev.tolist(),'folds':repeat.tolist()})
 write(ROOT/'evidence/confirmation_split.json',{'development_rows':b.dev.tolist(),'confirmation_rows':np.load(bank/'confirmation_rows.npy').tolist(),'labels_sealed_until_terminal_assessment':True})
 x=b.numeric[b.dev[::20]];write(ROOT/'evidence/eda.json',{'bank_id':b.manifest['bank_id'],'rows':len(b.y),'fold_counts':np.bincount(b.folds).tolist(),'positive_rate':float(b.y.mean()),'missing_rate_sample':float(np.isnan(x).mean()),'extra_features':b.extra.shape[1],'point_in_time_assertions_passed':True,'test_labels_accessed':False})
 from scipy.stats import ks_2samp
 from utils import sha,ks
 manifest=read(bank/'manifest.json')
 for name,checksum in manifest['raw_sha256'].items():
  rawfile=ROOT/'raw'/(name+'.csv');assert sha(rawfile)==checksum,('Raw-source hash changed',name)
 trainx=b.numeric[b.dev[::20]];testrows=np.flatnonzero(b.metadata.split.to_numpy()=='test');testx=b.numeric[testrows[::20]];source_coverage={}
 for source in ['repayment','bureau','account','shopping','previous']:
  cols=[j for j,n in enumerate(b.names) if source in b.manifest['info'][n]['sources']][:40]
  source_coverage[source]=float(np.isfinite(trainx[:,cols]).any(axis=1).mean()) if cols else None
 drift=[]
 for j in range(min(64,trainx.shape[1])):
  a=trainx[:,j];v=testx[:,j];a=a[np.isfinite(a)];v=v[np.isfinite(v)]
  if len(a) and len(v):drift.append({'feature':b.names[j],'train_test_ks':float(ks_2samp(a,v).statistic)})
 np.savez_compressed(ROOT/'evidence/distributions.npz',quantiles=np.nanquantile(trainx,[.01,.1,.5,.9,.99],axis=0),missing=np.isnan(trainx).mean(0))
 write(ROOT/'evidence/eda_details.json',{'source_coverage_sample':source_coverage,'train_test_drift_sample':drift,'decision_months':b.metadata.decision_datetime.iloc[b.dev].astype(str).str[:7].value_counts().to_dict(),'source_hashes_verified':True,'label_boundary':'Development labels only; drift uses unlabeled covariates'})
 (ROOT/'reserved').mkdir(exist_ok=True);np.save(ROOT/'reserved/confirmation_rows.npy',np.load(bank/'confirmation_rows.npy'));np.save(ROOT/'reserved/confirmation_y.npy',np.load(bank/'confirmation_y.npy'))
 from supervisor import rpc,remote,SSH
 payload=ROOT/'features-transfer.tar.gz'
 with tarfile.open(payload,'w:gz') as t:
  for sub in ['extras','evidence']:
   for file in (ROOT/sub).rglob('*'):
    if file.is_file():t.add(file,arcname=str(file.relative_to(ROOT)),recursive=False)
 rpc({'action':'put','destination':'incoming'},payload=payload)
 # Receipt is atomic: an incomplete feature transfer is never visible as ready.
 rpc({'action':'activate_features'})
 subprocess.run(['rsync','-az','--exclude=manifest.json','-e',shlex.join(SSH+['-p','2003']),str(ROOT/'extras'),str(ROOT/'evidence'),'faris@gpu-labkc.ub.ac.id:'+cfg['hosts']['pc3']['root']+'/'],check=True)
 subprocess.run(['rsync','-a','-e',shlex.join(SSH+['-p','2003']),str(ROOT/'extras/manifest.json'),'faris@gpu-labkc.ub.ac.id:'+cfg['hosts']['pc3']['root']+'/extras/manifest.json'],check=True)
 if not (ROOT/'features-distributed.json').exists():write(ROOT/'features-distributed.json',{'manifest_sha256':sha(ROOT/'extras/manifest.json'),'time':time.time()})
# PC1/3 receive generated features after PC2 preprocessing. Node payload includes them.
while not (ROOT/'extras/manifest.json').exists():
 if time.time()>deadline:raise TimeoutError('Shared campaign deadline')
 time.sleep(3)
for name,checksum in read(ROOT/'extras/manifest.json')['files'].items():assert sha(ROOT/'extras'/name)==checksum,('Feature transfer hash mismatch',name)
subprocess.run([sys.executable,str(ROOT/'host_prepare.py'),host],check=True)
if host=='node380':
 c=read(ROOT/'campaign/config.json')
 # host_prepare freezes node-specific resources/weights before seal; no post-seal edits.
 os.environ.update(HF_HOME='/home/taqiyudinmiftah/faris/kumo-large-node380-20261016/hf_cache',HF_HUB_OFFLINE='1')
action='resume' if (ROOT/'campaign/state.json').exists() else 'run'
os.execv(sys.executable,[sys.executable,str(ROOT/'campaign/runtime/campaign.py'),action,str(ROOT/'campaign')])
