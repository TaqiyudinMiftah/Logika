"""Regenerate frozen meta recipes, replay their CV outputs, then export one CSV."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import json
import numpy as np
import pandas as pd
from common import R,P,A,read,sha,dump,utc,sources
from gating import calibrate,calibrated,fit_stack,stacked

def main(replay_only=False):
 config=read(R/'CONFIG.json');sample=pd.read_csv(A/'exploration-12h-20261017/raw/sample_submission(3).csv')
 assert len(sample)==30000 and sample.application_id.is_unique
 reports=[];final=[];anchors=[]
 for fold in range(5):
  anchor_root=R/f'private/anchor/fold{fold}';anchor_report=read(anchor_root/'complete.json')
  assert sha(anchor_root/'test_predictions.npz')==anchor_report['files']['test_predictions.npz']
  a=np.load(anchor_root/'test_predictions.npz');anchor_test=pd.Series(a['prediction'],index=a['application_id']).reindex(sample.application_id).to_numpy()
  anchors.append(anchor_test)
  roles=np.load(P/f'packets/fold{fold}/roles.npz');anchor=np.load(P/f'packets/fold{fold}/anchor.npz');yc=roles['y_correction'];assignment=anchor['inner_fold']
  for seed in config['seeds']:
   task=f'kumo_large_global99_sub25k_e2__s{seed}__f{fold}';attempt=1;old=P/f'runs/{task}/attempt{attempt}'
   old_result=read(old/'RESULT.json');assert sha(old/'prediction.npz')==old_result['prediction_sha256']
   raw=np.load(old/'prediction.npz')['raw'];inner=np.full(6000,np.nan)
   for block in range(3):
    train,heldout=np.flatnonzero(assignment!=block),np.flatnonzero(assignment==block)
    inner[heldout]=calibrated(calibrate(raw[train],yc[train]),raw[heldout])
   full_theta=calibrate(raw[:6000],yc);stack_theta,audit=fit_stack(anchor['correction_oof'],inner,yc,penalty=.03)
   replay=stacked(stack_theta,anchor['validation'],calibrated(full_theta,raw[6000:]))
   original=P/f'composed/kumo_large_global99_sub25k_e2/s{seed}/fold{fold}'
   assert sha(original/'prediction.npz')==read(original/'RESULT.json')['prediction_sha256']
   error=float(np.max(np.abs(replay-np.load(original/'prediction.npz')['stack'])));assert error<1e-12,('Frozen postprocessing replay',seed,fold,error)
   if replay_only:
    reports.append(dict(fold=fold,seed=seed,postprocessing_replay_max_abs=error))
    continue
   root=R/f'predictions/s{seed}/fold{fold}';result=read(root/'RESULT.json');assert sha(root/'prediction.npz')==result['prediction_sha256']
   data=np.load(root/'prediction.npz');test_raw=pd.Series(data['raw'],index=data['application_id']).reindex(sample.application_id).to_numpy()
   prediction=stacked(stack_theta,anchor_test,calibrated(full_theta,test_raw));assert np.isfinite(prediction).all()
   final.append(prediction);reports.append(dict(fold=fold,seed=seed,postprocessing_replay_max_abs=error,
        raw_prediction_sha256=sha(root/'prediction.npz'),old_raw_sha256=sha(old/'prediction.npz'),
        original_stack_result_sha256=sha(original/'RESULT.json'),anchor_sha256=sha(anchor_root/'test_predictions.npz'),
        meta_rows=6000,weights_saved=False,stack=audit))
 anchor_mean=np.mean(anchors,axis=0)
 expected=pd.read_csv(R/'private/anchor/submission_B_maturity_only_full_20261025.csv')
 assert expected.application_id.tolist()==sample.application_id.tolist()
 assert sha(R/'private/anchor/submission_B_maturity_only_full_20261025.csv')=='a3c2884102f5ca5705cfcdc9a0bd025d558055d334edf31a2d543e8f40d58b57'
 assert np.max(np.abs(anchor_mean-expected.default_probability.to_numpy()))<1e-15
 if replay_only:
  dump(dict(status='PASS',UTC=utc(),members=reports,original_calibration_and_stacking_replayed=True,
       baseline_B_fivefold_CSV_replayed=True,source_sha256=sources()),R/'CALIBRATION_REPLAY_AUDIT.json')
  print(json.dumps(dict(status='PASS',original_members_replayed=len(reports))),flush=True)
  return
 probability=np.mean(final,axis=0);assert probability.shape==(30000,) and np.isfinite(probability).all()
 assert np.all((probability>0)&(probability<1)) and np.std(probability)>0
 output=R/'submission_KumoLarge99_sub25k_e2_stack_KS05312164.csv';assert not output.exists()
 pd.DataFrame(dict(application_id=sample.application_id,default_probability=probability)).to_csv(output,index=False)
 readback=pd.read_csv(output);assert list(readback)==list(sample)
 assert readback.application_id.tolist()==sample.application_id.tolist() and np.max(np.abs(readback.default_probability.to_numpy()-probability))<1e-15
 dump(dict(status='PASS',UTC=utc(),submission=output.name,sha256=sha(output),rows=30000,
      selected_development_KS=config['development_KS'],new_refit_validation_KS=None,public_score=None,
      members=reports,source_sha256=sources(),context_rows_per_fit=25000,available_training_rows=70000,feature_views=5,seeds=config['seeds'],
      original_postprocessing_replayed=True,anchor_B_replayed=True,uploaded=False,
      independent_confirmation=False,fitted_weights_saved=False,KV_caches_saved=False),R/'SUBMISSION_AUDIT.json')
 print(json.dumps(dict(status='PASS',sha256=sha(output),submission=str(output))),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--replay-only',action='store_true');args=parser.parse_args();main(args.replay_only)
