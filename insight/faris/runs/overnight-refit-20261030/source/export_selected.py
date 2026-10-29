"""Replay a frozen study's exact four-mode meta recipe before fresh export."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd
from gating import calibrate,calibrated,fit_stack,stacked
R=Path(__file__).resolve().parents[1];A=Path('/mnt/home/faris/logika-submit-best-sub25k-20261029/private/anchor')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,default=str)+'\n');q.replace(p)
def main(namespace):
 selection=json.load(open(R/'selected_fullfits'/namespace/'SELECTION.json'));study=R/namespace;design=selection['design'];mode=selection['mode'];kind=selection['kind'];predictions=[];reports=[]
 for seed in [2026,7731]:
  for fold in range(5):
   roles=np.load(study/f'packets/fold{fold}/roles.npz');a=np.load(study/f'packets/fold{fold}/anchor.npz');yc=roles['y_correction'];assign=a['inner_fold'];old=study/f'runs/{design}__s{seed}__f{fold}/attempt1';v=json.load(open(old/'RESULT.json'));assert sha(old/'prediction.npz')==v['prediction_sha256'];raw=np.load(old/'prediction.npz')['raw'];inner=np.full(6000,np.nan)
   for b in range(3):
    train,held=np.flatnonzero(assign!=b),np.flatnonzero(assign==b);inner[held]=calibrated(calibrate(raw[train],yc[train]),raw[held])
   theta=calibrate(raw[:6000],yc);candidate=calibrated(theta,raw[6000:]);ar=A/f'fold{fold}';v=json.load(open(ar/'complete.json'));assert sha(ar/'test_predictions.npz')==v['files']['test_predictions.npz'];z=np.load(ar/'test_predictions.npz');fresh=R/f'predictions/{kind}/s{seed}/fold{fold}';rv=json.load(open(fresh/'RESULT.json'));assert rv['fresh_training']and rv['campaign']=='overnight-refit-20261030';assert sha(fresh/'prediction.npz')==rv['prediction_sha256']and sha(fresh/'START.json')==rv['start_sha256'];q=np.load(fresh/'prediction.npz');ids=q['application_id'].astype(str);anchor_test=pd.Series(z['prediction'],index=z['application_id'].astype(str)).reindex(ids).to_numpy();test=calibrated(theta,q['raw'])
   if mode=='stack':
    stack,_=fit_stack(a['correction_oof'],inner,yc,penalty=.03);replay=stacked(stack,a['validation'],candidate);prediction=stacked(stack,anchor_test,test)
   elif mode=='standalone':replay=candidate;prediction=test
   else:
    w=.1 if mode=='fixed_010'else json.load(open(study/f'composed/{design}/s{seed}/fold{fold}/FROZEN.json'))['selected_meta_weight'];assert mode in ['fixed_010','meta_adaptive'];replay=(1-w)*a['validation']+w*candidate;prediction=(1-w)*anchor_test+w*test
   ref=study/f'composed/{design}/s{seed}/fold{fold}';assert sha(ref/'prediction.npz')==json.load(open(ref/'RESULT.json'))['prediction_sha256'];error=float(np.max(abs(replay-np.load(ref/'prediction.npz')[mode])));assert error<1e-12
   predictions.append(prediction);reports.append(dict(seed=seed,fold=fold,fresh_sha256=rv['prediction_sha256'],original_meta_replay_error=error))
 sample=pd.read_csv('/mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/raw/sample_submission(3).csv');p=pd.Series(np.mean(predictions,axis=0),index=ids).reindex(sample.application_id.astype(str)).to_numpy();assert np.isfinite(p).all()and np.all((p>0)&(p<1));out=R/f'submission_CURRENT_{namespace}.csv';pd.DataFrame(dict(application_id=sample.application_id,default_probability=p)).to_csv(out,index=False);rb=pd.read_csv(out);assert rb.application_id.tolist()==sample.application_id.tolist()and np.max(abs(rb.default_probability-p))<1e-15
 audit=R/f'AUDIT_CURRENT_{namespace}.json';dump(audit,dict(status='PASS',sha256=sha(out),rows=30000,selection=selection,members=reports,fresh_training=True,campaign='overnight-refit-20261030',new_refit_validation_KS=None,independent_confirmation=False))
 current=json.load(open(R/'READY_CANDIDATE.json'))
 if selection['KS']>current['local_KS']:dump(R/'READY_CANDIDATE.json',dict(path=str(out),audit_path=str(audit),sha256=sha(out),identity=namespace+'/'+selection['cohort'],local_KS=selection['KS'],campaign='overnight-refit-20261030',fresh_training=True))
if __name__=='__main__':main(sys.argv[1])
