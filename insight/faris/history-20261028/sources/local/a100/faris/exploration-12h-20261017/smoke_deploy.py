"""Synthetic detached SSH survival and automatic-report check."""
import sys,subprocess,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'));from utils import write
r=ROOT/'deployment-smoke';r.mkdir(exist_ok=True);s=r/'source';s.mkdir(exist_ok=True)
(s/'adapter.py').write_text('''import numpy as np,joblib
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp
def run_trial(ctx):
 x=np.random.default_rng(ctx.seed).normal(size=(256,8));y=(x[:,0]+x[:,1]>0).astype(int);m=LogisticRegression(C=ctx.params['C']).fit(x[:192],y[:192]);joblib.dump(m,ctx.path('model.joblib'));ctx.model('model.joblib');np.savez_compressed(ctx.path('validation.npz'),y=y[192:],prediction=m.predict_proba(x[192:])[:,1]);return {'validation':['validation.npz']}
def evaluate(ctx,result):
 z=np.load(ctx.path('validation.npz'));p=z['prediction'];y=z['y'];return {'score':float(ks_2samp(p[y==0],p[y==1]).statistic)}
''')
(s/'requirements.lock').write_text(''.join(n+'=='+importlib.metadata.version(n)+'\n' for n in ['numpy','scikit-learn','joblib','optuna','psutil','scipy']))
for n in ['rules.md','research.md','eda.json','hypotheses.md','applicability.md','split.json']:(r/n).write_text('{}' if n.endswith('json') else 'Synthetic infrastructure test only; no competition inputs.\n')
c={'version':1,'id':'explore12h-synthetic-smoke','competition_root':str(r),'adapter':'adapter.py','metric':'KS','direction':'maximize','inputs':{'data':str(r/'split.json')},'brief':{'description':'Synthetic SSH disconnect test','rules':str(r/'rules.md'),'compute_access':'PC2 own processes','external_data_policy':'None','pretrained_policy':'None'},'evidence':{n:str(r/(n+'.json' if n=='eda' else n+'.md')) for n in ['research','eda','hypotheses','applicability']},'protocols':{'screen':{'split':str(r/'split.json')}},'families':[{'name':'cpu','hypothesis':'Detached workers save actual models and automatic reports','parameters':{'C':{'type':'categorical','choices':[.01,.1,1.,10.]}},'resources':{'cpus':1,'ram_mb':600,'disk_mb':2,'timeout_seconds':10}}],'budget_seconds':25,'storage_limit_mb':100,'max_trials':4,'max_workers':2,'seeds':[2026],'reserve_fraction':.2,'breadth_fraction':.3,'poll_seconds':.2}
write(r/'input.json',c);subprocess.run([sys.executable,str(ROOT/'toolkit/campaign.py'),'bundle','--config',str(r/'input.json'),'--source',str(s),'--output',str(r/'campaign')],check=True);subprocess.run([sys.executable,str(r/'campaign/runtime/campaign.py'),'run',str(r/'campaign'),'--detach'],check=True)
