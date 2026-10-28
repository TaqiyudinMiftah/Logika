"""Terminal evaluation only, after allfive complete fitted pipelines exist."""
from pathlib import Path
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
import json,time,numpy as np
from scipy.stats import ks_2samp,rankdata
from sklearn.metrics import log_loss
import baseline as b
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
assert read(R/'STATUS.json')['status']=='training_complete_ready_for_PC2_scoring'
assert all(read(R/f'folds/fold{f}/complete.json')['status']=='complete'for f in range(5))
# Previously staged label-bearing reference is opened only in this terminal process.
z=np.load(R.parent/'tabicl-compact-a100-20261021/reference.npz');y=z['target'].astype(np.int64);fold=z['fold'];pred=dict(control90=z['control90'],quantile90=np.full(70000,np.nan),classifier90_A100=np.full(70000,np.nan));seen=np.zeros(70000,int)
for f in range(5):
 d=R/f'folds/fold{f}';c=read(d/'complete.json')
 for n,h in c['files'].items():assert b.sha(d/n)==h
 cache=read(d/'cache.json');assert cache['full_replay_max_error']<2e-5 and(d/'fitted_context.pt').stat().st_size==cache['bytes'];assert cache['sha256']==c['retained_cache_sha256']
 a=np.load(d/'validation.npz');assert set(a['rows'])==set(np.flatnonzero(fold==f));pred['quantile90'][a['rows']]=a['prediction'];seen[a['rows']]+=1
 ref=R.parent/f'kumo-full-context-20261021/folds/fold{f}';complete=read(ref/'complete.json');assert b.sha(ref/'validation.npz')==complete['files']['validation.npz'];other=np.load(ref/'validation.npz');assert np.array_equal(other['rows'],a['rows'])and np.array_equal(other['application_id'],a['application_id']);pred['classifier90_A100'][a['rows']]=other['context50000']
assert np.all(seen==1)
pred['fixed20percent']=.8*pred['control90']+.2*pred['quantile90']
def metric(y,p):
 assert np.isfinite(p).all();m=b.scores(y,p);assert abs(m['KS']-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12;n1=int(y.sum());n0=len(y)-n1;auc=float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0));assert abs(auc-m['AUC'])<1e-12;m['logloss']=float(log_loss(y,p));return m
out=dict(status='point_metrics_complete',metrics={k:metric(y,p)for k,p in pred.items()},fold_metrics={k:[metric(y[fold==f],p[fold==f])for f in range(5)]for k,p in pred.items()},rows=70000,goal_point_reached=any(metric(y,pred[k])['KS']>=.54 for k in ['quantile90','fixed20percent']),historical_label_reuse=True,submission_generated=False,scope='One preregisteredquantilepriorpipeline and fixed20percentblend. HardwarematchedA100classifier90 is a diagnostic repeat, not new scoringdesign. Corrections differ with rawprior bydesign; this doesnotisolate pretrainedweights alone.')
np.savez_compressed(R/'OOF.npz',target=y,fold=fold,**pred);b.dump(out,R/'POINT_RESULT.json')
rng=np.random.default_rng(20261116);groups=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in[0,1]];contrasts=[('quantile90','classifier90_A100'),('fixed20percent','control90')];boot={a+'_vs_'+c:[]for a,c in contrasts}
for _ in range(2000):
 assert time.time()<read(R/'plan.json')['original_deadline'],'Originalcompute deadline; pointmetrics retained'
 ix=np.concatenate([rng.choice(g,len(g),replace=True)for g in groups]);m={k:b.scores(y[ix],p[ix])for k,p in pred.items()}
 for a,c in contrasts:boot[a+'_vs_'+c].append([m[a]['KS']-m[c]['KS'],m[a]['AUC']-m[c]['AUC']])
out.update(status='complete',conditional_intervals={k:np.quantile(v,[.00625,.99375],axis=0).tolist()for k,v in boot.items()},interval_scope='2000pairedfold/classbootstrap;twofrozencontrasts/twometricsBonferroni. Conditionalonpredictions, nothistoricalsearch/retraininguncertainty.');b.dump(out,R/'result.json')
