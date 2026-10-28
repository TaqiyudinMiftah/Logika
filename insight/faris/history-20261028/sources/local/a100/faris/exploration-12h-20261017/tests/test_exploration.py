import sys,time,json,tempfile,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from utils import ks,combine

def test_shared_clock_is_not_reset(factory):
 from oneshot.controller import Controller
 from oneshot.common import atomic
 now=time.time();root=factory(budget_seconds=20,max_trials=1);clock=root.parent/'shared.json';atomic(clock,{'start':now-5,'deadline':now+4});config=json.loads((root/'config.json').read_text())
 # Re-bundle fixture with external-clock control to avoid editing a sealed bundle.
 source=root.parent/'clock-source';source.mkdir();(source/'adapter.py').write_text((root/'snapshots'/json.loads((root/'seal.json').read_text())['snapshot']/'code/adapter.py').read_text());(source/'requirements.lock').write_text((root/'snapshots'/json.loads((root/'seal.json').read_text())['snapshot']/'code/requirements.lock').read_text())
 config['external_clock']=str(clock);config['competition_root']=str(root.parent);inputpath=root.parent/'clock-input.json';atomic(inputpath,config)
 from oneshot.common import bundle
 out=root.parent/'clock-campaign';bundle(inputpath,source,out);controller=Controller(out);assert controller.state['start']==now-5;assert controller.state['deadline']==now+4;controller.run();assert json.loads((out/'state.json').read_text())['deadline']==now+4

def test_forced_rpc_refuses_path_escape():
 script=Path(__file__).resolve().parents[1]/'rpc.py'
 p=subprocess.run([sys.executable,str(script)],input=json.dumps({'action':'put','destination':'../outside'})+'\n',text=True,capture_output=True)
 assert p.returncode!=0

def test_invalid_ensemble_scores_are_rejected():
 try:ks([0,1],[.2,float('nan')])
 except AssertionError:pass
 else:raise AssertionError('Accepted NaN')


def test_nested_no_outer_label_access(monkeypatch):
 import nested,adapter
 class Bank:
  dev=np.arange(24);y=np.array([0,1]*12);folds=np.repeat(np.arange(3),8)
 bank=Bank();configs={'a':{'model':'lgb','view':'v','weight':1},'b':{'model':'elastic','view':'v','weight':1}};views={'v':{'config':{}}}
 monkeypatch.setattr(nested,'ExploreBank',lambda *a:bank)
 def read(path):
  if Path(path).name=='configurations.json':return configs
  if Path(path).name=='views.json':return views
  return {'folds':bank.folds.tolist()}
 monkeypatch.setattr(nested,'read',read);calls=[]
 def fit(ctx,b,p,rep,rows,y,query,folder,seed):
  assert not np.intersect1d(rows,query).size;np.testing.assert_array_equal(y,bank.y[rows]);calls.append((rows.copy(),query.copy(),folder));return np.repeat(.2+float(np.mean(y))*.4,len(query))
 monkeypatch.setattr(adapter,'fold_fit',fit)
 with tempfile.TemporaryDirectory() as temp:
  class Ctx:
   inputs={'bank':temp+'/bank','extras':temp};params={'config':'nested_0','components':['a','b']};seed=2026;spec={'protocol':'screen'};family='nested';split=Path(temp)/'split.json'
   def path(self,n):p=Path(temp)/n;p.parent.mkdir(parents=True,exist_ok=True);return p
   def model(self,n):assert self.path(n).stat().st_size>0
   def save_json(self,n,v):self.path(n).write_text(json.dumps(v))
  ctx=Ctx();nested.run(ctx);z=np.load(ctx.path('validation.npz'));assert len(z['prediction'])==24 and np.isfinite(z['prediction']).all();assert len(calls)==24

def test_archive_pruning_requires_matching_artifact_hash(monkeypatch,tmp_path):
 import importlib.util,hashlib
 script=Path(__file__).resolve().parents[1]/'archive_ops.py';spec=importlib.util.spec_from_file_location('archiveops',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);monkeypatch.setattr(module,'ROOT',tmp_path)
 folder=tmp_path/'campaign/runs/j/attempt_001';folder.mkdir(parents=True);(tmp_path/'logs').mkdir();p=folder/'model.joblib';p.write_bytes(b'retained model');checksum=hashlib.sha256(p.read_bytes()).hexdigest();(folder/'outcome.json').write_text(json.dumps({'status':'complete','models':['model.joblib'],'artifacts':{'model.joblib':checksum}}));request={'directory':'campaign/runs/j/attempt_001','archive':'verified-pc2-archive','files':{'model.joblib':'wrong'}}
 try:module.prune(request)
 except AssertionError:pass
 else:raise AssertionError('Pruned without verified matching bytes')
 assert p.exists();request['files']['model.joblib']=checksum;receipt=module.prune(request);assert not p.exists() and receipt['bytes_removed']>0 and (folder/'archive_receipt.json').exists();assert module.prune(request)['bytes_removed']==0
