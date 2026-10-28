from pathlib import Path
import time,json
import numpy as np
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def check():assert time.time()<read(R/'training-plan.json')['original_deadline']and not(R/'STOP').exists()
def metric(y,p):
 m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12;n=int(y.sum());assert abs((rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n))-m['AUC'])<1e-12;m['logloss']=float(log_loss(y,p));return m
def main():
 assert all((R/f'runs/fold{f}/complete.json').exists()for f in range(5));ref=np.load(R.parent/'maturity-placement-fe-20261021/OOF.npz');y=ref['target'].astype(np.int64);fold=ref['fold'];pr=np.full(70000,np.nan);rp=np.full(70000,np.nan);seen=np.zeros(70000,int)
 for f in range(5):
  d=R/f'runs/fold{f}'
  for n,h in read(d/'complete.json')['files'].items():assert b.sha(d/n)==h
  z=np.load(d/'validation.npz');ix=z['rows'];assert set(ix)==set(np.flatnonzero(fold==f));pr[ix]=z['prediction'];rp[ix]=z['replayed_prediction'];seen[ix]+=1
 assert np.all(seen==1)and np.isfinite(pr).all()and np.isfinite(rp).all()
 pred=dict(baseline=ref['baseline'],maturity_only=ref['both'],coverage_information=pr);mm={k:metric(y,p)for k,p in pred.items()};fm={k:[metric(y[fold==f],p[fold==f])for f in range(5)]for k,p in pred.items()};replay=metric(y,rp)
 out=dict(status='point_metrics_complete',metrics=mm,fold_metrics=fm,replayed_model_metrics=replay,goal_point_reached=mm['coverage_information']['KS']>=.54 and replay['KS']>=.54,confirmation_opened=False,historical_label_reuse=True,submission_generated=False)
 np.savez_compressed(R/'OOF.npz',target=y,fold=fold,replayed_prediction=rp,**pred);b.dump(out,R/'POINT_RESULT.json');print(json.dumps(out),flush=True)
 groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]];rng=np.random.default_rng(20261121);boot={k:[]for k in ['baseline','maturity_only']}
 for _ in range(1000):
  check();ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);m={k:b.scores(y[ix],p[ix])for k,p in pred.items()}
  for k in boot:boot[k].append([m['coverage_information']['KS']-m[k]['KS'],m['coverage_information']['AUC']-m[k]['AUC']])
 out.update(status='complete',intervals={k:np.quantile(v,[.05/8,1-.05/8],axis=0).tolist()for k,v in boot.items()},uncertainty='1000paired fold/class bootstrap,2contrasts x2metrics,Bonferroni95percent; conditional, no adaptive-selection or retraining correction.');b.dump(out,R/'result.json');np.savez_compressed(R/'bootstrap.npz',**boot)
if __name__=='__main__':main()
