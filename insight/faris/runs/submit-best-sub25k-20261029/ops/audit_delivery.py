"""Independently verify fixed member identities, sampled rows, source and CSV."""
from pathlib import Path
import sys,json,hashlib,datetime
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import numpy as np
import pandas as pd
from common import read,sha,dump,sources,P,A
from context_recipe import sample_context
from gating import calibrate,calibrated,fit_stack,stacked
cfg=read(R/'CONFIG.json');audit=read(R/'SUBMISSION_AUDIT.json');assert audit['status']=='PASS';expected={Path(k).name:v for k,v in read(R/'SOURCE_SEAL.json').items()};assert sources()==expected;sample=pd.read_csv(A/'exploration-12h-20261017/raw/sample_submission(3).csv');final=[];members=[]
for fold in range(5):
 packet=R/f'packets/fold{fold}';y=np.load(packet/'y_context.npy');m=read(packet/'MANIFEST.json');roles=np.load(P/f'packets/fold{fold}/roles.npz');anchor=np.load(P/f'packets/fold{fold}/anchor.npz');at=np.load(R/f'private/anchor/fold{fold}/test_predictions.npz');anchor_test=pd.Series(at['prediction'],index=at['application_id']).reindex(sample.application_id).to_numpy()
 for seed in cfg['seeds']:
  d=R/f'predictions/s{seed}/fold{fold}';s=read(d/'START.json');v=read(d/'RESULT.json');assert sha(d/'START.json')==v['start_sha256']and sha(d/'prediction.npz')==v['prediction_sha256'];assert s['source_sha256']==expected and s['parameters']==cfg['parameters']and s['context_rows']==25000 and s['available_training_rows']==70000;assert s['effective_seed']==seed+fold
  selected,draw=sample_context(y,25000,seed+fold,0);assert draw==s['context_draw_seed'];assert hashlib.sha256(np.asarray(selected,dtype='<i8').tobytes()).hexdigest()==s['selected_rows_sha256'];assert np.bincount(y[selected],minlength=2).tolist()==s['selected_class_counts'];assert v['audit']['first_batch_replay_error']<1e-7 and v['audit']['num_estimators']==2 and not v['fitted_weights_saved']and not v['KV_caches_saved'];assert s['versions']['structured-data-models']=='0.2.1'and s['versions']['torch'].startswith('2.7.1')
  raw=np.load(P/f'runs/{cfg["design"]}__s{seed}__f{fold}/attempt1/prediction.npz')['raw'];inner=np.full(6000,np.nan);assignment=anchor['inner_fold'];yc=roles['y_correction']
  for block in range(3):
   tr,va=np.flatnonzero(assignment!=block),np.flatnonzero(assignment==block);inner[va]=calibrated(calibrate(raw[tr],yc[tr]),raw[va])
  theta=calibrate(raw[:6000],yc);stack,_=fit_stack(anchor['correction_oof'],inner,yc,penalty=.03);replay=stacked(stack,anchor['validation'],calibrated(theta,raw[6000:]));ref=np.load(P/f'composed/{cfg["design"]}/s{seed}/fold{fold}/prediction.npz')['stack'];assert np.max(np.abs(replay-ref))<1e-12
  test=np.load(d/'prediction.npz');p=pd.Series(test['raw'],index=test['application_id']).reindex(sample.application_id).to_numpy();assert len(p)==30000 and np.isfinite(p).all();final.append(stacked(stack,anchor_test,calibrated(theta,p)));members.append(dict(seed=seed,fold=fold,result_sha256=sha(d/'RESULT.json'),sampled_rows_sha256=s['selected_rows_sha256']))
file=R/audit['submission'];table=pd.read_csv(file);assert list(table)==list(sample)and table.application_id.tolist()==sample.application_id.tolist()and table.application_id.is_unique;assert sha(file)==audit['sha256'];error=float(np.max(np.abs(table.default_probability.to_numpy()-np.mean(final,axis=0))));assert error<1e-15
old=pd.read_csv('/mnt/home/faris/logika-submit-highest-ks-20261029/submission_KumoLarge99_e2_stack_KS05312029.csv');assert old.application_id.tolist()==table.application_id.tolist();delta=np.abs(old.default_probability.to_numpy()-table.default_probability.to_numpy());assert delta.max()>0
value=dict(status='PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),submission=file.name,sha256=sha(file),rows=30000,members_verified=members,context_rows_each=25000,available_training_rows=70000,exact_sampled_rows_recomputed=True,full_source_identity_verified=True,original_meta_stack_replayed=True,CSV_reconstruction_max_abs=error,probability_min=float(table.default_probability.min()),probability_max=float(table.default_probability.max()),rows_different_from_previous_submission=int(np.sum(delta>0)),mean_absolute_difference_from_previous=float(delta.mean()),no_fitted_weights_or_KV_saved=True,selected_local_KS=cfg['development_KS'],new_refit_validation_KS=None)
dump(value,R/'DELIVERY_AUDIT.json');print(json.dumps({k:v for k,v in value.items()if k!='members_verified'}))
