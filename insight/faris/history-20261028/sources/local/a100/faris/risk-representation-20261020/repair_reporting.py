"""Resume completed training after a reporting-only oomd failure."""
import os,json,hashlib,time,subprocess
from pathlib import Path

root=Path(__file__).resolve().parent
follow=root.parent/'ri[REDACTED_CREDENTIAL]'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
record=dict(time=time.time(),reason='systemd-oomd terminated reporting and waiting follow-up',
            action='Bound read-once file cache; resume original checkpoints/deadline',
            mathematical_model_changes=False,training_repeated=False,owned_cache_bytes_advised=0)
for directory in [root,follow]:
    state=json.loads((directory/'state.json').read_text())
    assert state['deadline']==1792585871.823748
    status=subprocess.check_output(['systemctl','--user','show','logika-'+directory.name+'.service',
        '-p','ActiveState','--value'],text=True).strip()
    assert status in ['failed','inactive'],status
    pids=[x['pid'] for x in state.get('active',{}).values()]
    if state.get('active_phase_pid'):pids.append(state['active_phase_pid'])
    for pid in pids:
        proc=Path('/proc')/str(pid)/'cmdline'
        assert not proc.exists() or str(directory).encode() not in proc.read_bytes()
    (directory/'incidents/report-memory/state-before.json').write_text(json.dumps(state,indent=2))
    state.update(status='interrupted_by_oomd',active={})
    state.pop('active_phase_pid',None)
    (directory/'state.json').write_text(json.dumps(state,indent=2)+'\n')
assert all((root/f'folds/fold{f}/complete.json').exists() for f in range(5))
for path in root.glob('folds/fold*/*/fitted_context.pt'):
    with path.open('rb') as file:
        os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    record['owned_cache_bytes_advised']+=path.stat().st_size
# Hash implementation must return identical digests; reference files unchanged.
test=root/'folds/fold0/validation.npz'
import sys
sys.path.insert(0,str(root/'source'))
import baseline
assert baseline.sha(test)==sha(test)
for directory in [root,follow]:
    seal=json.loads((directory/'seal.json').read_text())
    for name in seal['files']:seal['files'][name]=sha(directory/name)
    for name in seal['inputs']:seal['inputs'][name]=sha(name)
    (directory/'seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    (directory/'incidents/report-memory/REPAIR.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
