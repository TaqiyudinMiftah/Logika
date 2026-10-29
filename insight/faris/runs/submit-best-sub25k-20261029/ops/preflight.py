from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import numpy as np
from common import read,dump,sha,utc,sources,P
OLD=Path('/mnt/home/faris/logika-submit-highest-ks-20261029');cfg=read(R/'CONFIG.json');assert cfg['cohort']==read(P/'FINAL_SELECTION.json')['identity']['cohort'];assert sha(P/'FINAL_SELECTION.json')==cfg['source_selection_sha256'];assert sha(P/'FINAL_EVALUATION.json')==cfg['source_evaluation_sha256'];assert read(P/'FINAL_ROOT_AUDIT.json')['status']=='PASS'
assert sha(R/'source/kumo_engine.py')==sha(P/'source/gpu_kumo_engine.py')=='f05ccade0bfead7dae94d80965f6d99b15a7a7c0fe5abe6173d9f1499aa74697'
assert sha(R/'source/context_recipe.py')==sha(P/'source/gpu_models.py');assert sha(R/'source/gating.py')==sha(P/'source/gating.py');assert sources()=={Path(k).name:v for k,v in read(R/'SOURCE_SEAL.json').items()}
reports=[]
for f in range(5):
 packet=R/f'packets/fold{f}';m=read(packet/'MANIFEST.json');dev=read(P/f'packets/fold{f}/MANIFEST.json');assert m['columns']==dev['views']['global99'];assert m['original_manifest_sha256']==sha(P/f'packets/fold{f}/MANIFEST.json')
 for name,key in [('features.parquet','feature_sha256'),('y_context.npy','labels_sha256'),('test_ids.npy','test_ids_sha256')]:assert sha(packet/name)==m[key]
 assert np.array_equal(np.load(packet/'y_context.npy'),np.load(P/'evaluation_inputs/target.npy'))
 a=R/f'private/anchor/fold{f}';ref=read(a/'complete.json');assert sha(a/'test_predictions.npz')==ref['files']['test_predictions.npz']
 reports.append(dict(fold=f,production_manifest_sha256=sha(packet/'MANIFEST.json'),production_features_sha256=m['feature_sha256'],development_manifest_sha256=sha(P/f'packets/fold{f}/MANIFEST.json'),frozen_columns_equal=True,all70k_label_order_verified=True,anchor_prediction_sha256=sha(a/'test_predictions.npz')))
from assemble import main
main(replay_only=True)
dump(dict(status='PASS',UTC=utc(),source_selection_sha256=cfg['source_selection_sha256'],source_config_sha256=sha(P/'CONFIG.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),folds=reports,calibration_replay_audit_sha256=sha(R/'CALIBRATION_REPLAY_AUDIT.json'),reuse='Unchanged100k-row production feature packets, frozen global99 lists and training-only70k cohort statistics; previous bit-exact original-membership parity proof retained',no_scientific_selection_or_feature_change=True),R/'PREFLIGHT_AUDIT.json');print('PASS: reused full-data packets match all5 frozen feature views; all10 selected postprocessing recipes replay exactly')
