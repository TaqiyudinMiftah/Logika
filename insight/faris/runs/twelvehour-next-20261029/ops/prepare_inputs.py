"""Run once on canonical node; preserve inherited inputs without regenerating."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parents[1];P=Path('/mnt/home/faris/logika-twelvehour-20261029')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n in ['private','logs','runs','evaluation_inputs','supplied_data']: (R/n).mkdir(exist_ok=True)
assert json.load(open(P/'ARTIFACT_CONSISTENCY_AUDIT.json'))['status']=='PASS'
if not(R/'packets').exists():(R/'packets').symlink_to(P/'packets',target_is_directory=True)
for n in ['baseline.npz','target.npy']:shutil.copy2(P/'evaluation_inputs'/n,R/'evaluation_inputs'/n)
winner=P/'evaluation/cohorts/kumo_large_global99_sub25k_e2__stack/prediction.npz';shutil.copy2(winner,R/'evaluation_inputs/previous.npz')
for n in ['train_applications.csv','test_applications.csv']:
 if not(R/'supplied_data'/n).exists():(R/'supplied_data'/n).symlink_to((P/'supplied_data'/n).resolve())
inputs=dict(parent_root=str(P),parent_source_seal_sha256=sha(P/'SOURCE_SEAL.json'),parent_gate_sha256=sha(P/'GATE.json'),parent_artifact_audit_sha256=sha(P/'ARTIFACT_CONSISTENCY_AUDIT.json'),packet_manifests={str(f):sha(R/f'packets/fold{f}/MANIFEST.json')for f in range(5)},baseline_sha256=sha(R/'evaluation_inputs/baseline.npz'),target_sha256=sha(R/'evaluation_inputs/target.npy'),previous_sha256=sha(R/'evaluation_inputs/previous.npz'),train_csv_sha256=sha(R/'supplied_data/train_applications.csv'),test_csv_sha256=sha(R/'supplied_data/test_applications.csv'),all70000_reused_development=True,no_new_features=True)
(R/'INPUTS.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n');print(json.dumps(inputs))
