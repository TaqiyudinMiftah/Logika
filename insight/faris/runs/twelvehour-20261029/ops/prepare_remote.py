from pathlib import Path
import hashlib,json,os,shutil,datetime
R=Path('/mnt/home/faris/logika-twelvehour-20261029');P=Path('/mnt/home/faris/logika-sixhour-broad-20261028')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
R.mkdir(exist_ok=True)
for name in ['runs','logs','private','evaluation_inputs','supplied_data']:(R/name).mkdir(exist_ok=True)
if not(R/'packets').exists():(R/'packets').symlink_to(P/'packets',target_is_directory=True)
for name in ['baseline.npz','target.npy']:shutil.copy2(P/'evaluation_inputs'/name,R/'evaluation_inputs'/name)
shutil.copy2(P/'evaluation/cohorts/kumo_large_global99_e2__stack/prediction.npz',R/'evaluation_inputs/previous.npz')
raw=Path('/mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/raw')
for n in ['train_applications.csv','test_applications.csv']:
 if not(R/'supplied_data'/n).exists():(R/'supplied_data'/n).symlink_to(raw/n)
inputs=dict(parent_root=str(P),parent_source_seal_sha256=sha(P/'SOURCE_SEAL.json'),parent_gate_sha256=sha(P/'GATE.json'),parent_artifact_audit_sha256=sha(P/'ARTIFACT_CONSISTENCY_AUDIT.json'),packet_manifests={str(f):sha(P/f'packets/fold{f}/MANIFEST.json')for f in range(5)},baseline_sha256=sha(R/'evaluation_inputs/baseline.npz'),target_sha256=sha(R/'evaluation_inputs/target.npy'),previous_sha256=sha(R/'evaluation_inputs/previous.npz'),train_csv_sha256=sha(R/'supplied_data/train_applications.csv'),test_csv_sha256=sha(R/'supplied_data/test_applications.csv'),all70000_reused_development=True,no_new_features=True)
dump(R/'INPUTS.json',inputs)
# Node2003 already holds a byte-identical parent packet copy; leave it read-only.
print(json.dumps(inputs))
