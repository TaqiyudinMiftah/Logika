"""Reconstruct one retained context on the first free GPU; inspect no labels."""
from pathlib import Path
import os,time,subprocess
import numpy as np
import baseline as b
from robustness import R,read

unit='logika-transfer-rolebag-gpu1-20261025.service'
jobs=[(o,j)for o in range(5)for j in range(5)][1::2]
deadline=read(R/'H010.json')['CV_deadline_epoch']
while True:
    state=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()
    if state=='inactive':break
    assert state!='failed','GPU1 worker failed; do not hide a missing fit behind a replay'
    assert time.time()<deadline,'No free GPU before H010 CV deadline'
    time.sleep(20)
assert all((R/f'role_bag/outer{o}/member{j}/complete.json').exists()for o,j in jobs)
root=R/'role_bag/outer0/member0';shared=R/'role_bag/outer0/shared'
output=R/'role_bag_replay_member0.npz'
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1'
subprocess.run(['/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python',str(R/'source/predict_indexed_kumo.py'),
                '--model',str(root/'kumo_model.joblib'),'--rows',str(root/'raw_predictions.npz'),'--row-key','rows',
                '--head',str(root/'ordinary_head.joblib'),'--spline',str(shared/'spline.joblib'),'--output',str(output)],env=env,check=True)
old=np.load(root/'raw_predictions.npz');pred=np.load(output);expected=np.load(root/'ordinary_validation.npz')
assert np.array_equal(pred['application_id'],old['application_id'])
raw_error=float(np.max(np.abs(pred['raw']-old['prediction'])))
pipeline_error=float(np.max(np.abs(pred['prediction'][6000:]-expected['prediction'])))
assert raw_error<2e-6 and pipeline_error<2e-6
b.dump(dict(status='passed',scope='Fresh reconstruction of outer0/member0 from indexedfeaturetable, labelsource, checkpoint andseed; no validation labels inspected',query_rows=20000,validation_tail_rows=14000,raw_max_abs_error=raw_error,pipeline_max_abs_error=pipeline_error,raw_bitwise_exact=bool(np.array_equal(pred['raw'],old['prediction'])),pipeline_bitwise_exact=bool(np.array_equal(pred['prediction'][6000:],expected['prediction'])),source_sha256=b.sha(__file__),prediction_entrypoint_sha256=b.sha(R/'source/predict_indexed_kumo.py')),R/'ROLE_BAG_CONTEXT_RECONSTRUCTION_AUDIT.json')
