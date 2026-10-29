"""Independent arithmetic check of frozen terminal selection and uncertainty.

All labels are already reused development. This verifies calculations, not
out-of-sample validity, and cannot change candidate selection or fitted models.
"""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import hashlib,json,sys,time
import numpy as np
from scipy.stats import ks_2samp
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import check_seal,sha,load,dump,utc
check_seal();f=load('FINAL_EVALUATION.json');selected=load('FINAL_SELECTION.json');snapshot=load('evaluation/FINAL_SCOREBOARD_SNAPSHOT.json');audit=load('ARTIFACT_CONSISTENCY_AUDIT.json');completion=load('COMPLETION.json');c=load('CLOCK.json')
assert audit['status']=='PASS'and audit['completed_fits']==completion['completed_fits']==1272 and audit['not_completed']==98
assert completion['audit_sha256']==sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json')and completion['final_evaluation_sha256']==sha(R/'FINAL_EVALUATION.json')
assert f['final_selection_sha256']==sha(R/'FINAL_SELECTION.json')and selected['source_scoreboard_sha256']==sha(R/'evaluation/FINAL_SCOREBOARD_SNAPSHOT.json')
winner=min(snapshot['candidates'],key=lambda k:(-snapshot['candidates'][k]['metrics']['KS'],k));assert winner==selected['identity']['cohort']==f['selected_cohort'];assert len(snapshot['complete_designs'])==127 and len(snapshot['candidates'])==508
p=R/f'evaluation/cohorts/{winner}/prediction.npz';assert sha(p)==f['prediction_sha256']==selected['identity']['prediction_sha256'];y=np.load(R/'evaluation_inputs/target.npy');a=np.load(R/'evaluation_inputs/baseline.npz');anchor=a['prediction'];fold=a['fold'];prior=np.load(R/'evaluation_inputs/previous.npz')['prediction'];pred=np.load(p)['prediction']
ks=lambda z:float(ks_2samp(z[y==0],z[y==1],method='asymp').statistic)
assert abs(ks(pred)-snapshot['candidates'][winner]['metrics']['KS'])<1e-12
strata=[np.flatnonzero((fold==i)&(y==j))for i in range(5)for j in[0,1]];out={}
for name,reference in [('anchor',anchor),('prior',prior)]:
 rng=np.random.default_rng(20261029);orders=[np.argsort(z,kind='stable')for z in[pred,reference]];ends=[np.flatnonzero(np.r_[z[o][1:]!=z[o][:-1],True])for z,o in zip([pred,reference],orders)];difference=[];n1=float(y.sum());n0=float(len(y)-y.sum())
 for b in range(2000):
  assert time.time()<c['deadline_epoch']-30
  rows=np.concatenate([rng.choice(s,len(s),replace=True)for s in strata]);counts=np.bincount(rows,minlength=len(y));v=counts*(y/n1-(1-y)/n0);scores=[np.max(np.abs(np.cumsum(v[o])[e]))for o,e in zip(orders,ends)];difference.append(float(scores[0]-scores[1]))
 interval=np.quantile(difference,[.025,.975]);expected=f['uncertainty'if name=='anchor'else'previous_winner_uncertainty'];assert np.allclose(interval,expected['percentile95'],rtol=0,atol=1e-14);assert abs(ks(pred)-ks(reference)-expected['delta_KS'])<1e-12;out[name]={'delta_KS':ks(pred)-ks(reference),'percentile95':interval.tolist(),'replicates':2000,'seed':20261029}
assert f['primary_development_gain_supported']is False and f['local_target_054_point_reached']is False
flags=['BACKEND_COLLECTION_account'+str(i)+'_COMPLETE.json'for i in range(1,7)]+['BACKEND_COLLECTION_host119_COMPLETE.json'];assert all(load(n)['collected']for n in flags);assert all(load(n)['released']for n in flags[:-1]);resources=load('FINAL_RESOURCE_AUDIT.json');assert resources['all_owned_sessions_collected_and_released']and all(x['assignments']==0 and x['rate_CU_hour']==0 for x in resources['accounts'])
dump(dict(status='PASS',UTC=utc(),completed_fits=1272,complete_designs=127,cohorts=508,unattempted_resource_limited_fits=98,archived_preSTART_failures=12,primary_KS=ks(pred),baseline_KS=ks(anchor),previous_KS=ks(prior),selected_cohort=winner,largest_complete_cohort_rule_verified=True,conditional_interval_replays=out,final_evaluation_sha256=sha(R/'FINAL_EVALUATION.json'),selection_sha256=sha(R/'FINAL_SELECTION.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),clock_sha256=sha(R/'CLOCK.json'),resource_audit_sha256=sha(R/'FINAL_RESOURCE_AUDIT.json'),independent_statistical_confirmation=False,no_refit_or_submission=True),R/'FINAL_ROOT_AUDIT.json')
print('PASS: immutable selection,1272fits/508cohorts and both2000-replicate intervals independently replayed; all owned VMs released')
