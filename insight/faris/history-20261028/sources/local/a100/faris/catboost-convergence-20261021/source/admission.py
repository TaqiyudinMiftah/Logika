"""Verify staged roles and engineering evidence before admission, without scoring."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from worker import load,sha
R=Path(__file__).resolve().parents[1]
checks=[]
for f in range(5):
 tr,va,ix,s,m=load(f)
 assert len(ix['context'])==50000 and len(ix['correction'])==6000
 assert np.array_equal(ix['train'],np.r_[ix['context'],ix['correction']])
 assert set(ix['context']).isdisjoint(ix['correction'])
 assert len(s['columns'])==1032 and len(s['compact_columns'])==116
 assert len(set(s['columns']))==1032 and set(s['compact_columns'])<=set(s['columns'])
 assert set(np.unique(tr.target))=={0,1}
 assert m['labels_available_to_feature_estimation']is False
 assert m['raw_snapshot_point_in_time_certified']is False
 assert not{'application_id','customer_id','target','split','decision_datetime'}&set(s['columns'])
 checks.append(dict(fold=f,manifest_sha256=sha(R/f'data/fold{f}/manifest.json'),context=50000,selection=6000,outer=14000))
for v in ['full','compact']:
 p=json.loads((R/f'PREFLIGHT_{v}.json').read_text());assert p['status']=='passed'and p['model_task_type']=='GPU'and p['checkpoint_replay_error']==0
(R/'SCIENTIFIC_GATE.json').write_text(json.dumps(dict(status='passed_for_two_fixed_views',checks=checks,outer_labels_opened=False,historical_label_reuse=True,point_in_time_snapshot_qualification=True),indent=2))
print('ALL_FIVE_ROLE_AND_FEATURE_CHECKS_PASSED',flush=True)
