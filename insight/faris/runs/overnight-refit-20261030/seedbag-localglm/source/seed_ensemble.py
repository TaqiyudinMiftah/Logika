"""Locked2/4/8seed means of the original meta-adaptive member recipe."""
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
from common import R,load,sha,dump,ids_sha,check_seal,utc
from gating import calibrate,calibrated
P=R.parent;OLD=P/'localglm-cann';DESIGN='localglm_actuarial99_intercept_w64_r0p1_raw';A=Path('/mnt/home/faris/logika-submit-best-sub25k-20261029/private/anchor')
def root(seed):return OLD if seed in [2026,7731]else R
def available(seed):
 return all((root(seed)/f'composed/{DESIGN}/s{seed}/fold{f}/RESULT.json').exists()and(P/f'predictions/selected_localglm-cann/s{seed}/fold{f}/RESULT.json').exists()for f in range(5))
def member(seed,fold):
 d=root(seed);roles=np.load(d/f'packets/fold{fold}/roles.npz');a=np.load(d/f'packets/fold{fold}/anchor.npz');rawdir=d/f'runs/{DESIGN}__s{seed}__f{fold}/attempt1';rv=json.load(open(rawdir/'RESULT.json'));assert sha(rawdir/'prediction.npz')==rv['prediction_sha256'];raw=np.load(rawdir/'prediction.npz')['raw'];theta=calibrate(raw[:6000],roles['y_correction']);candidate=calibrated(theta,raw[6000:])
 composed=d/f'composed/{DESIGN}/s{seed}/fold{fold}';v=json.load(open(composed/'RESULT.json'));assert sha(composed/'FROZEN.json')==v['frozen_sha256']and sha(composed/'prediction.npz')==v['prediction_sha256'];fr=json.load(open(composed/'FROZEN.json'));w=fr['selected_meta_weight'];replay=(1-w)*a['validation']+w*candidate;assert np.max(abs(replay-np.load(composed/'prediction.npz')['meta_adaptive']))<1e-12
 fresh=P/f'predictions/selected_localglm-cann/s{seed}/fold{fold}';v=json.load(open(fresh/'RESULT.json'));st=json.load(open(fresh/'START.json'));assert v['campaign']=='overnight-refit-20261030'and v['fresh_training']and v['base_seed']==seed and v['fold']==fold;assert sha(fresh/'START.json')==v['start_sha256']and sha(fresh/'prediction.npz')==v['prediction_sha256'];z=np.load(fresh/'prediction.npz');ar=A/f'fold{fold}';av=json.load(open(ar/'complete.json'));assert sha(ar/'test_predictions.npz')==av['files']['test_predictions.npz'];anchor=np.load(ar/'test_predictions.npz');ids=z['application_id'].astype(str);anchor_test=pd.Series(anchor['prediction'],index=anchor['application_id'].astype(str)).reindex(ids).to_numpy();test=(1-w)*anchor_test+w*calibrated(theta,z['raw'])
 return roles['validation'],replay,ids,test,dict(seed=seed,fold=fold,raw_sha256=rv['prediction_sha256'],fresh_sha256=v['prediction_sha256'],fresh_START_sha256=v['start_sha256'],composed_sha256=sha(composed/'prediction.npz'),meta_weight=w,member_validation_replay_max_error=float(np.max(abs(replay-np.load(composed/'prediction.npz')['meta_adaptive']))))
def evaluate():
 check_seal();seedsets=load('SEED_SETS.json');y=np.load(R/'evaluation_inputs/target.npy');board=[];allmembers={};fullproof=[]
 for seed in sorted(set(sum(seedsets.values(),[]))):
  if not available(seed):continue
  q=np.full(70000,np.nan);coverage=np.zeros(70000,int)
  for f in range(5):
   rows,p,ids,test,proof=member(seed,f);q[rows]=p;coverage[rows]+=1;fullproof.append(proof)
  assert np.all(coverage==1);allmembers[seed]=q
 for name,seeds in seedsets.items():
  if not all(s in allmembers for s in seeds):continue
  p=np.mean([allmembers[s]for s in seeds],axis=0);d=R/'evaluation'/name;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'prediction.npz',rows=np.arange(70000),prediction=p);dump(dict(seeds=seeds,source_seal_sha256=sha(R/'SOURCE_SEAL.json'),prediction_sha256=sha(d/'prediction.npz'),mode='meta_adaptive',frozen_before_new_outer_score=True),d/'FROZEN.json')
  k=float(ks_2samp(p[y==1],p[y==0]).statistic);v=dict(id=name,KS=k,seeds=seeds,mode='meta_adaptive',prediction_sha256=sha(d/'prediction.npz'));dump(v,d/'RESULT.json');board.append(v)
 assert any(x['id']=='seedbag2'and abs(x['KS']-.5316414268134972)<1e-12 for x in board),'Exact control replay failed'
 board.sort(key=lambda x:(-x['KS'],x['id']));selected=board[0];dump(dict(candidates=board,selected=selected,rule='highest complete audited point; fixed2/4/8seed cohorts; no split/architecture search',adaptive_development_only=True),R/'FROZEN_SEED_SELECTION.json')
 counts={k:len(list((R/'runs').glob('*/attempt1/'+f)))for k,f in [('completed_fits','RESULT.json'),('failure_records','FAILURE.json'),('interruption_records','INTERRUPTION.json')]}
 for f in(R/'runs').glob('*/attempt1/RESULT.json'):
  v=json.load(open(f));st=json.load(open(f.parent/'START.json'));assert sha(f.parent/'prediction.npz')==v['prediction_sha256']and sha(f.parent/'START.json')==v['start_sha256'];assert st['source_seal_sha256']==sha(R/'SOURCE_SEAL.json');assert not st['outer_labels_loaded']and v['audit']['batch_replay_max_abs_probability_error']<=5e-6
 audit=dict(status='PASS',UTC=utc(),scope='Raw/source/hash/fold coverage, original2seed exactKS, all eligible cohort means and fullproduction member/meta replay',counts=counts,eligible_seedsets=[v['id']for v in board],member_records=fullproof,source_seal_sha256=sha(R/'SOURCE_SEAL.json'),independent_confirmation=False)
 dump(audit,R/'ARTIFACT_CONSISTENCY_AUDIT.json');dump(dict(status='complete_adaptive_development_assessment',selected_cohort=selected['id']+'__meta_adaptive',primary_result=dict(metrics=dict(KS=selected['KS'])),candidates=board,independent_confirmation=False,additional_development_fits=counts['completed_fits'],additional_production_fits=sum(s not in [2026,7731]for s in allmembers)*5),R/'FINAL_EVALUATION.json');return selected
def export():
 selected=load('FROZEN_SEED_SELECTION.json')['selected'];seeds=selected['seeds'];pred=[];proof=[]
 for seed in seeds:
  for f in range(5):
   rows,validation,ids,p,record=member(seed,f);pred.append(p);proof.append(record)
 sample=pd.read_csv('/mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/raw/sample_submission(3).csv');p=pd.Series(np.mean(pred,axis=0),index=ids).reindex(sample.application_id.astype(str)).to_numpy();assert np.isfinite(p).all()and np.all((p>0)&(p<1));out=P/f'submission_CURRENT_{selected["id"]}_localglm.csv';pd.DataFrame(dict(application_id=sample.application_id,default_probability=p)).to_csv(out,index=False);rb=pd.read_csv(out);assert len(rb)==30000 and rb.application_id.tolist()==sample.application_id.tolist()and np.max(abs(rb.default_probability-p))<1e-15
 audit=P/'AUDIT_CURRENT_seedbag_localglm.json';dump(dict(status='PASS',campaign='overnight-refit-20261030',fresh_training=True,sha256=sha(out),candidate=selected,rows=30000,members=proof,source_seal_sha256=sha(R/'SOURCE_SEAL.json'),study_artifact_audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json'),new_refit_validation_KS=None,independent_confirmation=False),audit)
 current=json.load(open(P/'READY_CANDIDATE.json'))
 if selected['KS']>current['local_KS']:dump(dict(path=str(out),audit_path=str(audit),sha256=sha(out),identity=selected['id']+'_localglm',local_KS=selected['KS'],campaign='overnight-refit-20261030',fresh_training=True),P/'READY_CANDIDATE.json')
if __name__=='__main__':evaluate();export()
