"""Synthetic, metered-free acceptance checks. Never fits a competition model."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
import json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
from utils import write,ks,combine,sha

def run(host):
 start=time.time();proof={'host':host,'checks':{},'passed':False};rng=np.random.default_rng(9173);x=rng.normal(size=(180,12)).astype('float32');y=(x[:,0]+rng.normal(size=180)>.1).astype(int)
 for p in [np.zeros(len(y)),np.round(rng.random(len(y)),1),rng.random(len(y))]:assert abs(ks(y,p)-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
 proof['checks']['independent_tied_ks']=True
 arrays={'a':np.array([.1,.9]),'b':np.array([.5,.5]),'c':np.array([.8,.2])};np.testing.assert_array_equal(combine({'method':'trimmed','parents':list(arrays)},arrays),[.5,.5]);proof['checks']['trimmed_aggregation']=True
 from adapter import model
 for k in ['lgb','elastic','spline','extra','rff','ebm']:
  m=model({'model':k,'weight':1,'trees':5,'minimum':10,'strength':.05},9173,1,False);m.fit(x,y);assert np.isfinite(m.predict_proba(x)).all()
 proof['checks']['cpu_model_libraries']=True
 if host=='pc1':
  # Load the tested OpenCL library from the earlier dedicated AMD install if necessary.
  m=model({'model':'lgb','weight':1,'trees':5,'minimum':5},9173,1,False,True);m.fit(x,y);assert np.isfinite(m.predict_proba(x)).all();proof['checks']['rx6600_opencl_training']=True
 else:
  import torch
  from neural import make,train
  assert torch.cuda.is_available() and torch.cuda.device_count()==2
  proof['checks']['gpus']=[]
  class Ctx:
   spec={'resources':{'cpus':1,'gpus':1}}
   def __init__(self,r):self.root=Path(r);self.models=[]
   def check_stop(self):pass
   def path(self,n):p=self.root/n;p.parent.mkdir(parents=True,exist_ok=True);return p
   def model(self,n):assert self.path(n).exists();self.models.append(n)
   def save_json(self,n,v):write(self.path(n),v)
  for device in [0,1]:
   torch.cuda.set_device(device);a=torch.randn(512,512,device='cuda');(a@a).sum().backward() if a.requires_grad else None
   # Actual CUDA forward/backward/update, not just device enumeration.
   m=torch.nn.Linear(12,1).cuda();opt=torch.optim.AdamW(m.parameters());z=torch.tensor(x,device='cuda');loss=torch.nn.functional.binary_cross_entropy_with_logits(m(z).squeeze(),torch.tensor(y,dtype=torch.float32,device='cuda'));loss.backward();opt.step();torch.cuda.synchronize();proof['checks']['gpus'].append({'device':device,'name':torch.cuda.get_device_name(device),'loss':float(loss.detach()),'peak_bytes':torch.cuda.max_memory_allocated(device)})
  torch.cuda.set_device(0)
  with tempfile.TemporaryDirectory() as tmp:
   ctx=Ctx(tmp)
   for kind in ['tabm','resmlp','ft','tabr','gru','deepsets']:
    p={'width':128,'blocks':2,'dropout':.1,'lr':.0003,'weight':1,'encoding':'ple' if kind=='tabm' else 'plain'};events=(rng.normal(size=(len(x),8,9)).astype('float32'),np.repeat(8,len(x))) if kind in ['gru','deepsets'] else None
    pred,_=train(ctx,kind,x,y,x,'test_'+kind,p,events,events,epochs=1);assert pred.shape==(len(x),) and np.isfinite(pred).all()
   proof['checks']['neural_forward_backward_and_model_retention']=len(ctx.models)
  proof['checks']['all_four_nvidia_gpus_training_on_both_hosts']=True
 proof['passed']=True;proof['seconds']=time.time()-start;write(ROOT/'evidence'/('preflight-'+host+'.json'),proof);print(json.dumps(proof))

if __name__=='__main__':run(sys.argv[1])
