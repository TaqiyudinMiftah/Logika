"""Separate numerical implementation of the registered seed bag and conditional audit.

This verifies artifacts and arithmetic; it does not create independent assessment data.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ks(labels,scores):
    # Independently build both empirical CDFs at the pooled distinct thresholds.
    positive=np.sort(scores[labels==1]);negative=np.sort(scores[labels==0]);thresholds=np.unique(scores)
    return float(np.max(np.abs(np.searchsorted(positive,thresholds,side='right')/len(positive)-np.searchsorted(negative,thresholds,side='right')/len(negative))))
def calibration(raw,labels):
    z=logit(np.clip(np.asarray(raw,float),1e-6,1-1e-6));labels=np.asarray(labels,float)
    def fun(t):
        eta=t[0]+t[1]*z;e=expit(eta)-labels;d=t-[0.,1.]
        return float(np.mean(np.logaddexp(0.,eta)-labels*eta)+.005*np.dot(d,d)),np.array([np.mean(e),np.mean(e*z)])+.01*d
    q=minimize(fun,[0.,1.],jac=True,method='L-BFGS-B',bounds=[(-5.,5.),(0.,5.)],options=dict(maxiter=400,ftol=1e-12,gtol=1e-8));assert q.success
    return q.x
def apply(t,p):return expit(t[0]+t[1]*logit(np.clip(p,1e-6,1-1e-6)))
def getraw(kind,seed,fold,phase):
    if seed in [2026,7731]:return np.load(R/f'original/{kind}/s{seed}/fold{fold}/{phase}.npz')['raw']
    p=R/f'runs/{kind}__s{seed}__f{fold}/{phase}';j=json.loads((p/'RESULT.json').read_text());assert sha(p/'prediction.npz')==j['prediction_sha256']
    return np.load(p/'prediction.npz')['raw']
grid=np.array([0.,.025,.05,.1,.15,.2,.3,.4,.6,1.])
def rebuild(seeds):
    validation=np.zeros(70000);test_members=[];weights=[]
    for seed in seeds:
        for f in range(5):
            p=R/f'packets/fold{f}';roles=np.load(p/'roles.npz');a=np.load(p/'anchor.npz');yc=roles['y_correction'].astype(int);assignment=a['inner_fold'];anchor_test=np.load(p/'anchor_test.npy');branches={}
            for kind in ['localglm','cann','kumo70']:
                raw=getraw(kind,seed,f,'validation');production=getraw(kind,seed,f,'production');inner=np.empty(6000)
                for b in range(3):
                    tr=np.flatnonzero(assignment!=b);va=np.flatnonzero(assignment==b);inner[va]=apply(calibration(raw[tr],yc[tr]),raw[va])
                theta=calibration(raw[:6000],yc);branches[kind]=(inner,apply(theta,raw[6000:]),apply(theta,production))
            li,lv,lt=branches['localglm'];w=float(grid[int(np.argmax([ks(yc,(1-v)*a['correction_oof']+v*li) for v in grid]))]);weights.append(dict(seed=seed,fold=f,weight=w))
            x=np.column_stack([np.ones(6000)]+[logit(np.clip(v,1e-6,1-1e-6)) for v in [a['correction_oof'],branches['cann'][0],branches['kumo70'][0]]]);prior=np.array([0.,1.,0.,0.])
            def objective(t):
                eta=x@t;d=t-prior
                return float(np.mean(np.logaddexp(0.,eta)-yc*eta)+.5*.003*(d@d)),x.T@(expit(eta)-yc)/6000+.003*d
            q=minimize(objective,prior,jac=True,method='L-BFGS-B',bounds=[(-5,5)]+[(0,5)]*3,options=dict(maxiter=500,ftol=1e-12,gtol=1e-8));assert q.success
            def fused(v):return expit(np.column_stack([np.ones(len(v[0]))]+[logit(np.clip(z,1e-6,1-1e-6)) for z in v])@q.x)
            fv=fused([a['validation'],branches['cann'][1],branches['kumo70'][1]]);ft=fused([anchor_test,branches['cann'][2],branches['kumo70'][2]])
            validation[roles['validation']]+=(.9*((1-w)*a['validation']+w*lv)+.1*fv)/len(seeds)
            test_members.append(.9*((1-w)*anchor_test+w*lt)+.1*ft)
    return validation,np.mean(test_members,axis=0),weights
pre=json.loads((R/'PRE_EVALUATION_AUDIT.json').read_text());assert pre['status']=='PASS'
frozen=json.loads((R/'SEED_INCLUSION_FROZEN.json').read_text());available=pre['eligible_new_seeds'];assert frozen['eligible_new_seeds']==available
final=json.loads((R/'FINAL_EVALUATION.json').read_text());assert final['new_seeds']==available
assert json.loads((R/'GATE.json').read_text())['status']=='PASS'
oldv,oldt,_=rebuild([2026,7731]);reference=np.load(R/'evaluation_inputs/reference.npz');ids=np.load(R/'packets/fold0/test_ids.npy').astype(str)
rt=pd.Series(reference['test'],index=reference['ids'].astype(str)).reindex(ids).to_numpy()
oldcv_error=float(np.max(abs(oldv-reference['validation'])));oldtest_error=float(np.max(abs(oldt-rt)));assert oldcv_error<1e-12 and oldtest_error<1e-12
v,t,weights=rebuild([2026,7731]+available);saved=np.load(R/'evaluation/paired_equal_seedbag/prediction.npz')
cv_error=float(np.max(abs(v-saved['validation'])));test_error=float(np.max(abs(t-saved['test'])));assert cv_error<1e-12 and test_error<1e-12
assert np.array_equal(saved['ids'].astype(str),ids)
meta=json.loads((R/'evaluation/paired_equal_seedbag/FROZEN.json').read_text())
assert sha(R/'evaluation/paired_equal_seedbag/prediction.npz')==meta['prediction_sha256'] and sha(R/'evaluation/paired_equal_seedbag/submission_HELD.csv')==meta['CSV_sha256']
assert [x['weight'] for x in weights]==[x['LocalGLM_meta_weight'] for x in meta['meta']['records']]
sample=pd.read_csv(R/'sample_submission.csv');csv=pd.read_csv(R/'evaluation/paired_equal_seedbag/submission_HELD.csv',float_precision='round_trip')
assert list(csv.columns)==['application_id','default_probability'] and len(csv)==30000 and csv.application_id.nunique()==30000
assert np.array_equal(csv.application_id.astype(str),sample.application_id.astype(str))
aligned=pd.Series(saved['test'],index=ids).reindex(sample.application_id.astype(str)).to_numpy();assert np.array_equal(csv.default_probability.to_numpy(),aligned)
assert np.all((aligned>0)&(aligned<1))
# Only now open reused outer labels, after all complete predictions were frozen.
y=np.load(R/'evaluation_inputs/target.npy');baseline=np.load(R/'evaluation_inputs/baseline.npz')
# Recomposition was already checked against the original strict 1e-12 tolerance.
# Rank metrics and bootstrap audit the exact published bytes, including exact ties.
v=saved['validation']
base=baseline['validation'] if 'validation' in baseline.files else baseline['prediction'] if 'prediction' in baseline.files else baseline['pred']
baseks=ks(y,base);refks=ks(y,reference['validation']);value=ks(y,v)
assert abs(baseks-.5306361042400267)<1e-12 and abs(refks-.5319679799870825)<1e-12
assert abs(value-ks_2samp(v[y==1],v[y==0]).statistic)<1e-12
row=next(x for x in final['candidates'] if x['id']=='paired_equal_seedbag');assert abs(value-row['KS'])<1e-12
folds=np.empty(70000,int);deltas=[]
for f in range(5):
    ix=np.load(R/f'packets/fold{f}/roles.npz')['validation'];folds[ix]=f;deltas.append(ks(y[ix],v[ix])-ks(y[ix],reference['validation'][ix]))
assert np.max(abs(np.array(deltas)-row['fold_delta_reference']))<1e-12
# A separate conditional-bootstrap implementation, with the registered fixed RNG.
strata=[np.flatnonzero((folds==f)&(y==c)) for f in range(5) for c in [0,1]];rng=np.random.default_rng(20261031)
ordered=[]
for scores in [v,reference['validation']]:
    order=np.argsort(scores,kind='stable');end=np.flatnonzero(np.r_[scores[order][1:]!=scores[order][:-1],True]);ordered.append((order,end))
boot=[]
for _ in range(2000):
    draw=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata]);counts=np.bincount(draw,minlength=70000);increments=counts*(y/y.sum()-(1-y)/(len(y)-y.sum()))
    a,b=[np.max(np.abs(np.cumsum(increments[order])[end])) for order,end in ordered];boot.append(float(a-b))
ci=np.quantile(boot,[.025,.975]).tolist();assert np.max(abs(np.array(ci)-final['conditional_interval_reference']['percentile95']))<1e-12
selected=json.loads((R/'FINAL_SELECTION.json').read_text())['selected'];assert selected['id']=='paired_equal_seedbag'
supported=value-refks>=.0005 and ci[0]>0 and sum(x>0 for x in deltas)>=4
assert final['supported_gain']==supported and final['target054']==(value>=.54)
seed_metrics=[]
for seed in [2026,7731]+available:
    sv,_,_=rebuild([seed]);seed_metrics.append(dict(seed=seed,KS=ks(y,sv),AUC=float(roc_auc_score(y,sv)),diagnostic_only_after_primary_freeze=True))
report=dict(status='PASS',epoch=time.time(),new_seeds=available,total_averaged_seeds=2+len(available),baseline_KS=baseks,reference_KS=refks,selected_KS=value,selected_AUC=float(roc_auc_score(y,v)),reference_AUC=float(roc_auc_score(y,reference['validation'])),delta_reference=value-refks,conditional_interval_reference=ci,fold_delta_reference=deltas,seed_metrics=seed_metrics,independent_old_reference_CV_max_abs=oldcv_error,independent_old_reference_test_max_abs=oldtest_error,independent_candidate_CV_max_abs=cv_error,independent_candidate_test_max_abs=test_error,CSV_roundtrip_exact=True,CSV_sha256=meta['CSV_sha256'],bootstrap_replicates=2000,supported_gain=supported,target054=value>=.54,separate_arithmetic_implementation=True,independent_assessment_data=False,all70000_adaptively_reused=True,no_Kaggle_upload=True)
(R/'FINAL_ARITHMETIC_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
