"""Prospective held-meta fusion, plus audited export from fresh production members."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import ks_2samp
from gating import calibrate,calibrated
R=Path(__file__).resolve().parents[1];BASE=Path('/mnt/home/faris/logika-twelvehour-next-20261029');C=BASE/'neural-cann';OLD=Path('/mnt/home/faris/logika-twelvehour-20261029');SIX=Path('/mnt/home/faris/logika-sixhour-broad-20261028');ANCHOR=Path('/mnt/home/faris/logika-submit-best-sub25k-20261029/private/anchor');D='ncann_actuarial99_intercept_c0p1_w128_r0p0_lr0p0003'
BRANCHES={'cann':(C,D),'kumo25':(OLD,'kumo_large_global99_sub25k_e2'),'kumo70':(SIX,'kumo_large_global99_e2')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.load(open(p))
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,default=str)+'\n');t.replace(p)
def fit(x,y,penalty):
 prior=np.r_[0.,1.,np.zeros(x.shape[1]-2)]
 def loss(t):
  eta=x@t;d=t-prior;return float(np.mean(np.logaddexp(0.,eta)-y*eta)+.5*penalty*(d@d)),x.T@(expit(eta)-y)/len(y)+penalty*d
 q=minimize(loss,prior,jac=True,method='L-BFGS-B',bounds=[(-5,5)]+[(0,5)]*(x.shape[1]-1),options=dict(maxiter=500,ftol=1e-12,gtol=1e-8));assert q.success;return q.x

def member(kinds,penalty,seed,fold,production=False):
 roles=np.load(C/f'packets/fold{fold}/roles.npz');a=np.load(C/f'packets/fold{fold}/anchor.npz');yc=roles['y_correction'];assignment=a['inner_fold'];inner=[a['correction_oof']];outer=[a['validation']];test=[];evidence=[]
 if production:
  ar=ANCHOR/f'fold{fold}';v=read(ar/'complete.json');assert sha(ar/'test_predictions.npz')==v['files']['test_predictions.npz'];z=np.load(ar/'test_predictions.npz');ids=np.load(R/f'packets/fold{fold}/test_ids.npy');test.append(pd.Series(z['prediction'],index=z['application_id'].astype(str)).reindex(ids.astype(str)).to_numpy());assert np.isfinite(test[-1]).all()
 for kind in kinds:
  p,d=BRANCHES[kind];old=p/f'runs/{d}__s{seed}__f{fold}/attempt1';result=read(old/'RESULT.json');assert sha(old/'prediction.npz')==result['prediction_sha256'];raw=np.load(old/'prediction.npz')['raw'];assert len(raw)==20000
  r=np.load(p/f'packets/fold{fold}/roles.npz');assert np.array_equal(roles['validation'],r['validation'])and np.array_equal(roles['correction'],r['correction'])
  oof=np.full(6000,np.nan)
  for b in range(3):
   train,held=np.flatnonzero(assignment!=b),np.flatnonzero(assignment==b);oof[held]=calibrated(calibrate(raw[train],yc[train]),raw[held])
  theta=calibrate(raw[:6000],yc);inner.append(oof);outer.append(calibrated(theta,raw[6000:]));evidence.append(dict(kind=kind,development_raw_sha256=sha(old/'prediction.npz')))
  if production:
   fresh=R/f'predictions/{kind}/s{seed}/fold{fold}';v=read(fresh/'RESULT.json');st=read(fresh/'START.json');assert v['campaign']=='overnight-refit-20261030'and v['fresh_training']and st['fresh_training'];assert sha(fresh/'prediction.npz')==v['prediction_sha256']and sha(fresh/'START.json')==v['start_sha256'];assert v['fold']==fold and v['base_seed']==seed
   z=np.load(fresh/'prediction.npz');assert z['application_id'].astype(str).tolist()==ids.astype(str).tolist();test.append(calibrated(theta,z['raw']));evidence[-1]['fresh_prediction_sha256']=v['prediction_sha256']
 x=np.column_stack([np.ones(6000)]+[logit(np.clip(z,1e-6,1-1e-6))for z in inner]);theta=fit(x,yc,penalty)
 prediction=expit(np.column_stack([np.ones(14000)]+[logit(np.clip(z,1e-6,1-1e-6))for z in outer])@theta)
 if production:
  q=expit(np.column_stack([np.ones(30000)]+[logit(np.clip(z,1e-6,1-1e-6))for z in test])@theta);return q,evidence
 return prediction,evidence

def designs():
 v=[dict(id=k+'__stack',kinds=[k],penalty=.03)for k in BRANCHES]
 for k in ['kumo25','kumo70']:
  for p in [.003,.01,.03,.1]:v.append(dict(id='fusion_cann_'+k+'_p'+str(p),kinds=['cann',k],penalty=p))
 return v

def evaluate():
 frozen=R/'FROZEN_DEVELOPMENT_SELECTION.json'
 if frozen.exists():return read(frozen)
 y=np.load(C/'evaluation_inputs/target.npy');records=[]
 for design in designs():
  p=np.zeros(70000);coverage=np.zeros(70000,int);evidence=[]
  for seed in [2026,7731]:
   for f in range(5):
    q,ev=member(design['kinds'],design['penalty'],seed,f);rows=np.load(C/f'packets/fold{f}/roles.npz')['validation'];p[rows]+=q/2;coverage[rows]+=1;evidence+=ev
  assert np.all(coverage==2)and np.isfinite(p).all()
  # Predictions and source identities are frozen before their outer-label score.
  folder=R/'development'/design['id'];folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'prediction.npz',prediction=p);dump(folder/'FROZEN.json',dict(design=design,prediction_sha256=sha(folder/'prediction.npz'),source_sha256=sha(Path(__file__)),evidence=evidence))
  ks=float(ks_2samp(p[y==1],p[y==0]).statistic);record=dict(**design,KS=ks,prediction_sha256=sha(folder/'prediction.npz'),adaptive_development_only=True);records.append(record);dump(folder/'RESULT.json',record)
 records.sort(key=lambda x:(-x['KS'],x['id']));dump(frozen,dict(candidates=records,selection=records[0],rule='highest complete audited reused-development KS; lexical ties',new_public_score=None,independent_confirmation=False));return read(frozen)

def export():
 selection=evaluate();eligible=[]
 for d in selection['candidates']:
  if all((R/f'predictions/{k}/s{s}/fold{f}/RESULT.json').exists()for k in d['kinds']for s in [2026,7731]for f in range(5)):eligible.append(d)
 if not eligible:return
 selected=eligible[0];pred=[];evidence=[]
 for seed in [2026,7731]:
  for f in range(5):
   p,ev=member(selected['kinds'],selected['penalty'],seed,f,True);pred.append(p);evidence.append(dict(seed=seed,fold=f,branches=ev))
 probability=np.mean(pred,axis=0);assert np.isfinite(probability).all()and np.all((probability>0)&(probability<1));ids=np.load(R/'packets/fold0/test_ids.npy').astype(str)
 sample=pd.read_csv('/mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/raw/sample_submission(3).csv');probability=pd.Series(probability,index=ids).reindex(sample.application_id.astype(str)).to_numpy();assert np.isfinite(probability).all()
 out=R/f'submission_CURRENT_{selected["id"]}.csv';pd.DataFrame(dict(application_id=sample.application_id,default_probability=probability)).to_csv(out,index=False);rb=pd.read_csv(out);assert rb.application_id.tolist()==sample.application_id.tolist()and len(rb)==30000;assert np.max(abs(rb.default_probability-probability))<1e-15
 audit=R/'CURRENT_SUBMISSION_AUDIT.json';dump(audit,dict(status='PASS',campaign='overnight-refit-20261030',fresh_training=True,sha256=sha(out),candidate=selected,development_evidence_conditional=True,new_refit_validation_KS=None,rows=30000,members=evidence,selected_from_complete_current_refits=True,no_historical_prediction_fallback=True,preparation_audit_sha256=sha(R/'PREPARATION_AUDIT.json'),source_sha256=sha(Path(__file__))))
 dump(R/'READY_CANDIDATE.json',dict(campaign='overnight-refit-20261030',fresh_training=True,path=str(out),sha256=sha(out),audit_path=str(audit),identity=selected['id'],local_KS=selected['KS']))
 print(json.dumps(dict(status='CURRENT_TRAINED_CANDIDATE_READY',identity=selected['id'],KS=selected['KS'],sha256=sha(out))),flush=True)
if __name__=='__main__':export()
