"""Stage immutable matched inputs from PC2 to PC3, exclusively through PC2."""
from pathlib import Path
import json,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/mnt/home/faris/faris/kumo-coverage-placement-20261021'
def link(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.symlink_to(Path(source).resolve(),target_is_directory=Path(source).is_dir())
def main():
    plan=json.loads((ROOT/'plan.json').read_text());bundle=ROOT/'deployment/pc3';bundle.mkdir(parents=True,exist_ok=True)
    link(ROOT/'source',bundle/'source');link(plan['checkpoint'],bundle/'assets/classifier.pt')
    inp=Path(plan['input']);feature=Path(plan['feature_parent']);repair=Path(plan['repaired_features'])
    for name in ['BASELINE_LOCK.json','data/train_features.parquet']:link(inp/name,bundle/'input'/name)
    for f in range(5):
        for name in ['indices.npz','metadata.json']:link(inp/f'artifacts/fold{f}'/name,bundle/f'input/artifacts/fold{f}'/name)
        for name in ['features.parquet','complete.json']:link(repair/f'fold{f}'/name,bundle/f'repaired/fold{f}'/name)
    for name in ['kumo_oof.parquet','spline_oof.parquet']:link(inp/'artifacts/fold0'/name,bundle/'input/artifacts/fold0'/name)
    for name in ['INDEPENDENT_AUDIT.json','data/features.parquet']:link(feature/name,bundle/'coverage'/name)
    for name in ['fold0_correction.joblib','fold0_validation.parquet']:link(feature/'evidence/runs/V25'/name,bundle/'coverage/evidence/runs/V25'/name)
    for name in ['SCIENTIFIC_GATE.json','SOURCE_LOCK.json']:link(ROOT/name,bundle/name)
    for f in [3,4]:
        assert (ROOT/f'shared/fold{f}/complete.json').exists(),'Wait for shared spline completion; do not transfer partial models'
        link(ROOT/f'shared/fold{f}',bundle/f'shared/fold{f}')
    plan.update(input=REMOTE+'/input',checkpoint=REMOTE+'/assets/classifier.pt',feature_parent=REMOTE+'/coverage',
        repaired_features=REMOTE+'/repaired',assigned_folds=[3,4],python=REMOTE+'/.venv/bin/python')
    (bundle/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    subprocess.run(['rsync','-aL','-e','ssh -o BatchMode=yes -o ConnectTimeout=8 -p 2003',str(bundle)+'/',
        'faris@gpu-labkc.ub.ac.id:'+REMOTE+'/'],check=True)
    (ROOT/'PC3_STAGED.json').write_text(json.dumps(dict(status='complete',time=time.time(),root=REMOTE,
        checkpoint_sha256=plan['checkpoint_sha'],assigned_folds=[3,4],transfer='Immutable inputs copied; existing host environment and unrelated files preserved.'))+'\n')
if __name__=='__main__':main()
