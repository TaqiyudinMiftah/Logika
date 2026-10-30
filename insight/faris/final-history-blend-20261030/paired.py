"""Separate prospective recomposition of matched saved validation and fresh test members."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,json,time,hashlib,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.stats import ks_2samp,rankdata
R=Path(__file__).resolve().parent/'paired';OLD=Path('/mnt/home/faris/logika-overnight-refit-20261030');BASE=Path('/mnt/home/faris/logika-twelvehour-cross-20261030')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
assert (R/'PREPUBLICATION.json').exists()
sys.path.insert(0,str(OLD/'source'));import fusion
ids=np.load(OLD/'packets/fold0/test_ids.npy').astype(str);rows=np.arange(70000);bank={};failures=[]
for d in fusion.designs():
 if time.time()>1793364540:break
 try:
  output=R/'sources'/d['id'];q=[];evidence=[]
  for seed,f in itertools.product([2026,7731],range(5)):
   assert time.time()<1793364600
   p,ev=fusion.member(d['kinds'],d['penalty'],seed,f,True);q.append(p);evidence.append(dict(seed=seed,fold=f,branches=ev))
  cvpath=OLD/'development'/d['id']/'prediction.npz';z=np.load(cvpath);cv=z['prediction'];assert cv.shape==(70000,)
  test=np.mean(q,axis=0);assert np.isfinite(test).all()
  output.mkdir(parents=True,exist_ok=True);np.savez_compressed(output/'predictions.npz',validation=cv,test=test,ids=ids)
  dump(output/'FROZEN.json',dict(spec=d,source_sha256=sha(OLD/'source/fusion.py'),saved_CV_sha256=sha(cvpath),prediction_sha256=sha(output/'predictions.npz'),evidence=evidence,base_refitting=False))
  bank[d['id']]=(cv,test);print('reconstructed',d['id'],flush=True)
 except Exception as e:failures.append(dict(id=d['id'],error_type=type(e).__name__,error=str(e)[:300]));print('failed',d['id'],type(e).__name__,flush=True)
ready=json.loads((OLD/'READY_CANDIDATE.json').read_text());assert sha(ready['path'])==ready['sha256']
a=np.load(BASE/'evaluation_inputs/previous.npz');print('reference_keys',a.files,flush=True)
cv=a['prediction'];csv=pd.read_csv(ready['path']);test=pd.Series(csv.default_probability.to_numpy(),index=csv.application_id.astype(str)).reindex(ids).to_numpy();assert np.isfinite(test).all();bank['current']=(cv,test)
dump(R/'BANK_FROZEN.json',dict(sources=list(bank),failures=failures,epoch=time.time(),plan_sha256=sha(R/'PLAN.json'),source_sha256=sha(__file__),outer_targets_not_loaded=True))
candidates=dict(bank)
for n,(c,t) in bank.items():
 if n=='current':continue
 for w in [.1,.3,.5]:candidates[f'current_{n}_w{w}']=((1-w)*cv+w*c,(1-w)*test+w*t)
for mode in ['probability','logit','rank']:
 def aggregate(x):
  if mode=='probability':return np.mean(x,axis=0)
  if mode=='logit':return expit(np.mean(logit(np.clip(x,1e-6,1-1e-6)),axis=0))
  return np.mean([rankdata(v,method='average')/len(v) for v in x],axis=0)
 candidates['all_'+mode]=(aggregate([x[0] for x in bank.values()]),aggregate([x[1] for x in bank.values()]))
for n,(c,t) in candidates.items():
 p=R/'candidates'/n;p.mkdir(parents=True,exist_ok=True);np.savez_compressed(p/'predictions.npz',validation=c,test=t,ids=ids);pd.DataFrame(dict(application_id=ids,default_probability=t)).to_csv(p/'submission_HELD.csv',index=False);dump(p/'FROZEN.json',dict(prediction_sha256=sha(p/'predictions.npz'),CSV_sha256=sha(p/'submission_HELD.csv'),outer_targets_loaded=False,submitted=False))
dump(R/'ALL_PREDICTIONS_FROZEN.json',dict(count=len(candidates),epoch=time.time(),names=list(candidates)))
y=np.load(BASE/'evaluation_inputs/target.npy');records=[]
for n,(c,t) in candidates.items():
 assert np.isfinite(c).all() and c.shape==(70000,);score=float(ks_2samp(c[y==1],c[y==0]).statistic);records.append(dict(id=n,KS=score,delta_current=score-.5316414268134972,CSV_sha256=sha(R/'candidates'/n/'submission_HELD.csv')))
records.sort(key=lambda x:(-x['KS'],x['id']));dump(R/'RESULTS.json',dict(candidates=records,selected=records[0],failure_records=failures,all70000_reused_adaptive_development=True,no_untouched_audit=True,new_base_training=False,public_KS_unknown=True,submitted=False));print(json.dumps(records[:3]),flush=True)
