"""Independent arithmetic/hash audit of the sealed previous study; no selection changes."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.stats import ks_2samp
R=Path('/mnt/home/faris/logika-twelvehour-next-20261029');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.load(open(p))
assert(R/'COMPLETION.json').exists();plan=read(R/'UNION_PLAN.json');selection=read(R/'UNION_SELECTION.json');evaluation=read(R/'UNION_EVALUATION.json');y=np.load(R/'evaluation_inputs/target.npy');checked=[];total=0;fails=0;interruptions=0
for key,v in selection['candidates'].items():
 root=R if v['namespace']=='.'else R/v['namespace'];assert read(root/'ARTIFACT_CONSISTENCY_AUDIT.json')['status']=='PASS';assert sha(root/'FINAL_EVALUATION.json')==v['component_evaluation_sha256'];assert sha(root/'ARTIFACT_CONSISTENCY_AUDIT.json')==v['component_audit_sha256'];assert sha(root/'SOURCE_SEAL.json')==v['source_seal_sha256'];p=R/v['prediction'];assert sha(p)==v['prediction_sha256'];z=np.load(p)['prediction'];assert z.shape==(70000,)and np.isfinite(z).all();score=float(ks_2samp(z[y==1],z[y==0]).statistic);assert abs(score-v['KS'])<1e-12;checked.append(dict(key=key,KS=score,prediction_sha256=v['prediction_sha256']))
for namespace in plan['component_studies']:
 root=R if namespace=='.'else R/namespace;assert(root/'COMPLETION.json').exists();total+=len(list((root/'runs').glob('*/attempt1/RESULT.json')));fails+=len(list((root/'runs').glob('*/attempt1/FAILURE.json')));interruptions+=len(list((root/'runs').glob('*/attempt1/INTERRUPTION.json')))
winner=min(checked,key=lambda x:(-x['KS'],x['key']));assert winner['key']==selection['selected'];assert evaluation['selection_sha256']==sha(R/'UNION_SELECTION.json');baseline=np.load(R/'evaluation_inputs/baseline.npz')['prediction'];base=float(ks_2samp(baseline[y==1],baseline[y==0]).statistic);assert abs(base-.5306361042400267)<1e-12
out=dict(status='PASS',epoch=time.time(),source='independent SciPy/hash/max-selection replay',completed_fits=total,failed_records=fails,interruption_records=interruptions,registered_slots=11070,not_completed_slots=11070-total,checked_candidates=checked,selected=winner,baseline_KS=base,union_selection_sha256=sha(R/'UNION_SELECTION.json'),union_evaluation_sha256=sha(R/'UNION_EVALUATION.json'),clock_sha256=sha(R/'CLOCK.json'),new_training_or_selection_changes=False,all_case_data_adaptively_reused=True,independent_generalization_confirmation=False)
(R/'TERMINAL_UNION_REPLAY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='checked_candidates'}))
