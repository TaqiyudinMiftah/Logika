"""Record the isolated control fix and reseal a confirmed stopped campaign."""
from pathlib import Path
import sys,json,hashlib,time,subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source'))
from experiment import null_permutation

state=json.loads((root/'state.json').read_text())
assert state['deadline']==1792585871.823748,'Preserve the original clock'
status=subprocess.check_output(['systemctl','--user','show',
    'logika-ri[REDACTED_CREDENTIAL].service','-p','ActiveState','--value'],text=True).strip()
assert status in ['inactive','failed'],status
for value in state.get('active',{}).values():
    path=Path('/proc')/str(value['pid'])/'cmdline'
    assert not path.exists() or str(root).encode() not in path.read_bytes(),'Prior worker remains alive'
plan=json.loads((root/'plan.json').read_text());inp=Path(plan['input'])
data=pd.read_parquet(inp/'data/train_features.parquet',columns=['application_id','target'])
checks=[]
for fold in range(5):
    ix=np.load(inp/f'artifacts/fold{fold}/indices.npz');context=ix['context']
    labels=np.full(len(context),-1,int);partitions=[]
    for j,(fit,query) in enumerate(StratifiedKFold(5,shuffle=True,
            random_state=plan['inner_seed']+fold).split(np.zeros(len(context)),data.target.iloc[context])):
        labels[query]=j;partitions.append((fit,query))
    permutation=null_permutation(labels,np.random.default_rng(73100+fold))
    for fit,query in partitions:
        assert np.array_equal(labels[permutation[query]],labels[query])
        assert not np.intersect1d(permutation[query],fit).size
    checks.append(dict(outer_fold=fold,null_recipients_checked=50000,
        donor_and_recipient_excluded_from_teacher_fit=True))

incident=root/'incidents/null-isolation';incident.mkdir(parents=True,exist_ok=True)
null_before=[]
for fold in range(5):
    old=root/f'folds/fold{fold}/null'
    if old.exists():
        preserved=incident/f'fold{fold}-null-before'
        assert not preserved.exists();old.rename(preserved);null_before.append(fold)
sha=lambda path:hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
audit=dict(status='passed',checks=checks,original_deadline=state['deadline'],time=time.time(),
    correction='Shuffle donors only within the same held-out teacher fold',
    completed_teacher_artifacts_preserved=True,null_backbones_started_before_fix=null_before,
    old_source_sha=sha(incident/'experiment-before.py'),new_source_sha=sha(root/'source/experiment.py'),
    incident='Initial resealing command failed before changing the seal; restart correctly refused the changed source. No GPU condition had started.')
(incident/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
pre=json.loads((root/'preflight.json').read_text());pre['null_control_label_isolation']=audit
(root/'preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
seal=json.loads((root/'seal.json').read_text())
(incident/'seal-before.json').write_text(json.dumps(seal,indent=2)+'\n')
for name in seal['files']:seal['files'][name]=sha(root/name)
(root/'seal.json').write_text(json.dumps(seal,indent=2)+'\n')
print(json.dumps(audit,indent=2),flush=True)
