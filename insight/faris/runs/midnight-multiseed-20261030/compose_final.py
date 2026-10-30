"""Fixed branch composition; freeze seed inclusion/predictions before reused-outer scores."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,time,sys,itertools
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import logit,expit
from scipy.stats import ks_2samp
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'));from gating import calibrate,calibrated,fit_stack,stacked
GRID=[0.,.025,.05,.1,.15,.2,.3,.4,.6,1.]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def ks(y,p):
 y=np.asarray(y,int);order=np.argsort(p,kind='stable');ends=np.flatnonzero(np.r_[p[order][1:]!=p[order][:-1],True]);d=np.cumsum(y[order])/y.sum()-np.cumsum(1-y[order])/(1-y).sum();return float(np.max(abs(d[ends])))
def raw(kind,seed,f,phase):
 if seed in [2026,7731]and kind in ['localglm','cann','kumo70']:p=R/f'original/{kind}/s{seed}/fold{f}/{phase}.npz';z=np.load(p)
 else:
  d=R/f'runs/{kind}__s{seed}__f{f}/{phase}';j=json.loads((d/'RESULT.json').read_text());p=d/'prediction.npz';assert sha(p)==j['prediction_sha256']and sha(d/'START.json')==j['start_sha256'];z=np.load(p)
 return np.asarray(z['raw'],float)
def complete(kind,s):return all((R/f'runs/{kind}__s{s}__f{f}/{p}/RESULT.json').exists()for f,p in itertools.product(range(5),['validation','production']))
def branch(kind,s,f):
 p=R/f'packets/fold{f}';roles=np.load(p/'roles.npz');a=np.load(p/'anchor.npz');y=roles['y_correction'].astype(int);assign=a['inner_fold'];r=raw(kind,s,f,'validation');test=raw(kind,s,f,'production');assert len(r)==20000 and len(test)==30000;inner=np.full(6000,np.nan)
 for b in range(3):tr=np.flatnonzero(assign!=b);va=np.flatnonzero(assign==b);inner[va]=calibrated(calibrate(r[tr],y[tr]),r[va])
 theta=calibrate(r[:6000],y);outer=calibrated(theta,r[6000:]);test=calibrated(theta,test);w=GRID[int(np.argmax([ks(y,(1-v)*a['correction_oof']+v*inner)for v in GRID]))];anchor_test=np.load(p/'anchor_test.npy');st,_=fit_stack(a['correction_oof'],inner,y)
 return dict(inner=inner,outer=outer,test=test,adaptive_outer=(1-w)*a['validation']+w*outer,adaptive_test=(1-w)*anchor_test+w*test,stack_outer=stacked(st,a['validation'],outer),stack_test=stacked(st,anchor_test,test),w=w)
def fusion(c,k,f):
 p=R/f'packets/fold{f}';a=np.load(p/'anchor.npz');y=np.load(p/'roles.npz')['y_correction'].astype(float);x=np.column_stack([np.ones(6000)]+[logit(np.clip(v,1e-6,1-1e-6))for v in [a['correction_oof'],c['inner'],k['inner']]]);prior=np.array([0.,1.,0.,0.])
 def objective(t):eta=x@t;d=t-prior;return float(np.mean(np.logaddexp(0.,eta)-y*eta)+.5*.003*(d@d)),x.T@(expit(eta)-y)/6000+.003*d
 q=minimize(objective,prior,jac=True,method='L-BFGS-B',bounds=[(-5,5)]+[(0,5)]*3,options=dict(maxiter=500,ftol=1e-12,gtol=1e-8));assert q.success
 def predict(v):return expit(np.column_stack([np.ones(len(v[0]))]+[logit(np.clip(z,1e-6,1-1e-6))for z in v])@q.x)
 return predict([a['validation'],c['outer'],k['outer']]),predict([np.load(p/'anchor_test.npy'),c['test'],k['test']])
def paired(seeds):
 out=np.zeros(70000);test=[];coverage=np.zeros(70000,int);meta=[]
 for s,f in itertools.product(seeds,range(5)):
  l=branch('localglm',s,f);c=branch('cann',s,f);k=branch('kumo70',s,f);v,t=fusion(c,k,f);rows=np.load(R/f'packets/fold{f}/roles.npz')['validation'];out[rows]+=(.9*l['adaptive_outer']+.1*v)/len(seeds);coverage[rows]+=1;test.append(.9*l['adaptive_test']+.1*t);meta.append(dict(seed=s,fold=f,LocalGLM_meta_weight=l['w']))
 assert np.all(coverage==len(seeds));return out,np.mean(test,axis=0),meta
def save(name,val,test,meta):
 d=R/'evaluation'/name;d.mkdir(parents=True,exist_ok=True);ids=np.load(R/'packets/fold0/test_ids.npy').astype(str);np.savez_compressed(d/'prediction.npz',validation=val,test=test,ids=ids);sample=pd.read_csv(R/'sample_submission.csv');q=pd.Series(test,index=ids).reindex(sample.application_id.astype(str)).to_numpy();assert np.isfinite(q).all();pd.DataFrame(dict(application_id=sample.application_id,default_probability=q)).to_csv(d/'submission_HELD.csv',index=False,float_format='%.17g');dump(d/'FROZEN.json',dict(meta=meta,prediction_sha256=sha(d/'prediction.npz'),CSV_sha256=sha(d/'submission_HELD.csv'),outer_targets_loaded_for_this_new_score=False));return d
if __name__=='__main__':
 check='--preflight'in sys.argv
 if check:
  v,t,m=paired([2026,7731]);ref=np.load(R/'evaluation_inputs/reference.npz');assert np.max(abs(v-ref['validation']))<1e-12;oldids=ref['ids'].astype(str);ids=np.load(R/'packets/fold0/test_ids.npy').astype(str);r=pd.Series(ref['test'],index=oldids).reindex(ids).to_numpy();assert np.max(abs(t-r))<1e-12;dump(R/'COMPOSITION_REPLAY_GATE.json',dict(status='PASS',CV_max_abs=float(np.max(abs(v-ref['validation']))),test_max_abs=float(np.max(abs(t-r))),no_new_outer_scores=True));print('COMPOSITION_REPLAY_GATE PASS',flush=True);sys.exit()
 if (R/'FINAL_EVALUATION.json').exists():sys.exit()
 C=json.loads((R/'CONFIG.json').read_text());seeds=[s for s in C['new_seeds']if all(complete(k,s)for k in C['kinds'])];T=json.loads((R/'TRANSFORMER_COMPONENT.json').read_text());tsel={k:[s for s in T['seeds']if complete(k,s)]for k in T['kinds']};dump(R/'SEED_INCLUSION_FROZEN.json',dict(epoch=time.time(),eligible_new_seeds=seeds,excluded_new_seeds=[s for s in C['new_seeds']if s not in seeds],transformer_seeds=tsel,availability_only_no_seed_score_filter=True,source_sha256=sha(__file__)));candidates=[]
 v,t,m=paired([2026,7731]+seeds);candidates.append(save('paired_equal_seedbag',v,t,dict(old_seeds=[2026,7731],new_seeds=seeds,folds=5,records=m)))
 for k,ss in tsel.items():
  if len(ss)!=len(T['seeds']):continue
  values={mode:np.zeros(70000)for mode in ['adaptive','stack']};tests={mode:[]for mode in values}
  for s,f in itertools.product(ss,range(5)):
   a=branch(k,s,f);rows=np.load(R/f'packets/fold{f}/roles.npz')['validation']
   for mode in values:values[mode][rows]+=a[mode+'_outer']/len(ss);tests[mode].append(a[mode+'_test'])
  for mode in values:candidates.append(save(k+'__'+mode,values[mode],np.mean(tests[mode],axis=0),dict(seeds=ss,folds=5,head=k,mode=mode)))
 dump(R/'ALL_CANDIDATES_FROZEN.json',dict(candidates=[str(p.relative_to(R))for p in candidates],epoch=time.time()))
 y=np.load(R/'evaluation_inputs/target.npy');ref=np.load(R/'evaluation_inputs/reference.npz')['validation'];scores=[]
 for d in candidates:
  p=np.load(d/'prediction.npz')['validation'];score=float(ks_2samp(p[y==1],p[y==0]).statistic);fd=[]
  for f in range(5):ix=np.load(R/f'packets/fold{f}/roles.npz')['validation'];fd.append(float(ks(y[ix],p[ix])-ks(y[ix],ref[ix])))
  scores.append(dict(id=d.name,KS=score,delta_reference=score-.5319679799870825,fold_delta_reference=fd,prediction_sha256=sha(d/'prediction.npz'),CSV_sha256=sha(d/'submission_HELD.csv')))
 scores.sort(key=lambda q:(-q['KS'],q['id']));dump(R/'FINAL_SELECTION.json',dict(selected=scores[0],rule='LargestcompleteauditedrecipeKS,lexicalties;seedselectionavailabilityonly',frozen_before_conditional_intervals=True));sys.path.insert(0,str(R/'source'));from uncertainty import interval
 p=np.load(R/'evaluation'/scores[0]['id']/'prediction.npz')['validation'];foldids=np.empty(70000,int)
 for f in range(5):foldids[np.load(R/f'packets/fold{f}/roles.npz')['validation']]=f
 ci=interval(y,p,ref,foldids);dump(R/'FINAL_EVALUATION.json',dict(status='completed_adaptive_development',candidates=scores,new_seeds=seeds,conditional_interval_reference=ci,all70000_reused=True,no_untouched_audit=True,supported_gain=scores[0]['delta_reference']>=.0005 and ci['percentile95'][0]>0 and sum(x>0 for x in scores[0]['fold_delta_reference'])>=4,target054=scores[0]['KS']>=.54,no_Kaggle_submission=True));print(json.dumps(dict(selected=scores[0],interval=ci,new_seed_count=len(seeds))),flush=True)
