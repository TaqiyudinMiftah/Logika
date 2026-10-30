import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,time,subprocess,concurrent.futures,hashlib,importlib.util,functools
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent;OLD=Path('/mnt/home/faris/logika-midnight-multiseed-20261030');E=R/'ecommerce';PY='/mnt/home/faris/logika-model-arena-20261025/env311/bin/python';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(OLD/'source'));from gating import calibrate,calibrated
GRID=[0.,.025,.05,.1,.15,.2,.3,.4,.6,1.]
def ks(y,p):return float(ks_2samp(p[y==1],p[y==0]).statistic)
def collect():
 routes=[(['ssh','-p','2003'],'faris@gpu-labkc.ub.ac.id','/mnt/home/faris'),(['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119'],'faris@100.89.211.119','/home/faris'),(['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119'],'faris@100.98.169.42','/home/faris')]
 def one(route):
  ssh,host,home=route;d=R/'private/collected'/host.replace('@','_');d.mkdir(parents=True,exist_ok=True)
  subprocess.run(['rsync','-a','--exclude=*.tmp','-e',__import__('shlex').join(ssh),host+':'+home+'/logika-final-three-20261030/runs/',str(d)+'/runs/'],check=True,capture_output=True)
  subprocess.run([PY,str(OLD/'atomic_collect.py'),'--owner','finalthree','--source',str(d)],check=True,capture_output=True) if False else None
  for p in (d/'runs').glob('*/*'):
   if not(p/'RESULT.json').exists():continue
   j=json.loads((p/'RESULT.json').read_text());assert sha(p/'prediction.npz')==j['prediction_sha256'] and sha(p/'START.json')==j['start_sha256']
   dest=R/'runs'/p.relative_to(d/'runs')
   if(dest/'RESULT.json').exists():assert sha(dest/'RESULT.json')==sha(p/'RESULT.json');continue
   dest.mkdir(parents=True,exist_ok=True)
   for n in ['prediction.npz','START.json','RESULT.json']:__import__('shutil').copy2(p/n,dest/n)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3)as ex:list(ex.map(one,routes))
def complete(seed,ecommerce=False,phase='validation'):
 return all(((E if ecommerce else R)/f'runs/{k}__s{seed}__f{f}/{phase}/RESULT.json').exists()for k in (['localglm']if ecommerce else['localglm','cann','kumo70'])for f in range(5))
def raw(k,s,f,phase,ecommerce):
 if ecommerce and k!='localglm' or s in [2026,7731] and not ecommerce:
  return np.load(OLD/f'original/{k}/s{s}/fold{f}/{phase}.npz')['raw']
 root=E if ecommerce else R;p=root/f'runs/{k}__s{s}__f{f}/{phase}';j=json.loads((p/'RESULT.json').read_text());assert sha(p/'prediction.npz')==j['prediction_sha256'] and sha(p/'START.json')==j['start_sha256'];q=np.load(p/'prediction.npz');roles=np.load(OLD/f'packets/fold{f}/roles.npz');want=np.r_[roles['correction'],roles['validation']]if phase=='validation'else np.arange(70000,100000);assert np.array_equal(q['rows'],want) and np.isfinite(q['raw']).all();return q['raw']
@functools.lru_cache(None)
def member(k,s,f,ecommerce,with_test):
 p=OLD/f'packets/fold{f}';roles=np.load(p/'roles.npz');a=np.load(p/'anchor.npz');yc=roles['y_correction'].astype(int);v=raw(k,s,f,'validation',ecommerce);inner=np.empty(6000)
 for b in range(3):tr=np.flatnonzero(a['inner_fold']!=b);va=np.flatnonzero(a['inner_fold']==b);inner[va]=calibrated(calibrate(v[tr],yc[tr]),v[va])
 theta=calibrate(v[:6000],yc);out=calibrated(theta,v[6000:]);test=calibrated(theta,raw(k,s,f,'production',ecommerce))if with_test else None;return inner,out,test
def compose(s,ecommerce=False,with_test=False):
 out=np.empty(70000);tests=[]
 for f in range(5):
  p=OLD/f'packets/fold{f}';roles=np.load(p/'roles.npz');a=np.load(p/'anchor.npz');y=roles['y_correction'].astype(int);l=member('localglm',s,f,ecommerce,with_test);c=member('cann',s,f,ecommerce,with_test);k=member('kumo70',s,f,ecommerce,with_test)
  w=GRID[int(np.argmax([ks(y,(1-v)*a['correction_oof']+v*l[0])for v in GRID]))];x=np.column_stack([np.ones(6000)]+[logit(np.clip(z,1e-6,1-1e-6))for z in [a['correction_oof'],c[0],k[0]]]);prior=np.array([0.,1.,0.,0.])
  def obj(t):eta=x@t;d=t-prior;return float(np.mean(np.logaddexp(0.,eta)-y*eta)+.5*.003*(d@d)),x.T@(expit(eta)-y)/6000+.003*d
  q=minimize(obj,prior,jac=True,method='L-BFGS-B',bounds=[(-5,5)]+[(0,5)]*3,options=dict(maxiter=500,ftol=1e-12,gtol=1e-8));assert q.success
  def fusion(z):return expit(np.column_stack([np.ones(len(z[0]))]+[logit(np.clip(v,1e-6,1-1e-6))for v in z])@q.x)
  out[roles['validation']]=.9*((1-w)*a['validation']+w*l[1])+.1*fusion([a['validation'],c[1],k[1]])
  if with_test:
   anchor=np.load(p/'anchor_test.npy');tests.append(.9*((1-w)*anchor+w*l[2])+.1*fusion([anchor,c[2],k[2]]))
 return out,np.mean(tests,axis=0)if with_test else None
def stop_all():
 routes=[([],None),(['ssh','-p','2003'],'faris@gpu-labkc.ub.ac.id'),(['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119'],'faris@100.89.211.119'),(['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119'],'faris@100.98.169.42')]
 code="import subprocess,json; q=subprocess.run(['systemctl','--user','list-units','--no-legend','--plain','--state=running','logika-final-three-*'],capture_output=True,text=True); n=[l.split()[0] for l in q.stdout.splitlines() if l.strip() and 'evaluate'not in l and 'watchdog'not in l]; subprocess.run(['systemctl','--user','stop',*n],check=True) if n else None; print(json.dumps(n))"
 receipt=[]
 for ssh,host in routes:
  q=subprocess.run((ssh+[host,'python3 -c '+__import__('shlex').quote(code)])if host else['python3','-c',code],capture_output=True,text=True);receipt.append(dict(host=host or'canonical',exit_code=q.returncode,stopped=q.stdout,stderr=q.stderr))
 (R/'STOP_ALL_RECEIPT.json').write_text(json.dumps(dict(epoch=time.time(),owners=receipt,only_named_campaign_units=True),indent=2)+'\n')
results=[];done=set();deadline=1793379360.
try:
 while time.time()<deadline-30:
  collect()
  for name,s,ec in [('ecommerce',2026,True),('seed3',3,False),('seed4',4,False)]:
   if name in done or not complete(s,ec):continue
   v,t=compose(s,ec,ec and complete(s,True,'production'));d=R/'evaluation'/name;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'validation.npz',validation=v);(d/'FROZEN.json').write_text(json.dumps(dict(epoch=time.time(),seed=s,validation_sha256=sha(d/'validation.npz'),recipe_frozen_before_outer_score=True))+'\n')
   y=np.load(OLD/'evaluation_inputs/target.npy');ref=np.load(OLD/'evaluation_inputs/reference.npz')['validation'];score=ks(y,v);row=dict(id=name,seed=s,KS=score,AUC=float(roc_auc_score(y,v)),delta_reference=score-ks(y,ref),complete_five_fold_validation=True,production_complete=t is not None,all70000_adaptively_reused=True)
   if ec:
    matched,_=compose(2026,False,False);row['matched_seed2026_KS']=ks(y,matched);row['matched_feature_delta']=score-row['matched_seed2026_KS']
   if t is not None:
    ids=np.load(OLD/'packets/fold0/test_ids.npy').astype(str);sample=pd.read_csv(OLD/'sample_submission.csv');pred=pd.Series(t,index=ids).reindex(sample.application_id.astype(str)).to_numpy();assert np.isfinite(pred).all();pd.DataFrame(dict(application_id=sample.application_id,default_probability=pred)).to_csv(d/'submission_HELD.csv',index=False,float_format='%.17g');row['CSV_sha256']=sha(d/'submission_HELD.csv')
   results.append(row);done.add(name);(R/'RESULTS.json').write_text(json.dumps(dict(epoch=time.time(),experiments=results,reference_KS=.5319679799870825,exactly_three_registered=True),indent=2)+'\n');print(json.dumps(row),flush=True)
  if len(done)==3:break
  time.sleep(8)
finally:
 stop_all();(R/'FINAL.json').write_text(json.dumps(dict(epoch=time.time(),experiments=results,incomplete_experiments=sorted({'ecommerce','seed3','seed4'}-done),all_experiment_workers_cancelled=True,no_extra_experiments=True,no_Kaggle_upload=True,previous_reference_KS=.5319679799870825),indent=2)+'\n');print('FINAL_THREE_STOPPED',flush=True)
