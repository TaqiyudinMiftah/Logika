"""Prepare immutable per-fold roles on PC2; outer labels stay with its evaluator."""
from pathlib import Path
import json,hashlib,shutil,tarfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT.parent/'kumo-coverage-placement-20261021'
EXTRA=['risk_conditional_negative_months','risk_conditional_nonnegative_months',
    'aux_account_months','aux_orders90','aux_ecommerce_observed_days','aux_allocation_count']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    plan=json.loads((MAIN/'plan.json').read_text());inp=Path(plan['input'])
    assert json.loads((MAIN/'SPLINE_AUDIT.json').read_text())['status']=='passed'
    original=pd.read_parquet(inp/'data/train_features.parquet',columns=['application_id','target'])
    lock=json.loads((inp/'BASELINE_LOCK.json').read_text());columns=lock['Kumo']['features']+EXTRA
    head=lock['correction']['features']+EXTRA;needed=list(dict.fromkeys(columns+head))
    coverage=pd.read_parquet(Path(plan['feature_parent'])/'data/features.parquet')
    assert coverage.application_id.tolist()==original.application_id.tolist()
    dest=ROOT/'full_data';dest.mkdir(exist_ok=True)
    (dest/'head_columns.json').write_text(json.dumps(head)+'\n')
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'data/columns.json').write_text(json.dumps(columns)+'\n')
    for f in range(5):
        d=dest/f'fold{f}';assert not d.exists(),'Preserve previous immutable staged inputs';d.mkdir()
        repair=Path(plan['repaired_features'])/f'fold{f}'
        repaired=pd.read_parquet(repair/'features.parquet').set_index('application_id').loc[original.application_id].reset_index()
        for name in EXTRA:repaired[name]=coverage[name].to_numpy()
        ix=dict(np.load(inp/f'artifacts/fold{f}/indices.npz'));ci,hi,vi=ix['context'],ix['correction'],ix['validation']
        context=repaired.iloc[ci][['application_id']+columns].reset_index(drop=True)
        context['target']=original.target.iloc[ci].to_numpy(np.int8)
        context.to_parquet(d/'context.parquet',index=False)
        query=repaired.iloc[np.r_[hi,vi]][['application_id']+needed].reset_index(drop=True)
        assert 'target'not in query and not set(context.application_id)&set(query.application_id)
        query.to_parquet(d/'query.parquet',index=False)
        np.save(d/'correction_labels.npy',original.target.iloc[hi].to_numpy(np.int8))
        np.savez_compressed(d/'indices.npz',context=ci,correction=hi,validation=vi)
        shutil.copyfile(MAIN/f'shared/fold{f}/validation.npz',d/'spline_validation.npz')
        ci_hash=hashlib.sha256('\n'.join(context.application_id).encode()).hexdigest()
        assert ci_hash==json.loads((repair/'complete.json').read_text())['fit_ids_sha256']
        manifest=dict(fold=f,context_ids_sha256=ci_hash,context_rows=50000,correction_rows=6000,
            validation_rows=14000,outer_validation_target_column_staged=False,
            files={p.name:sha(p)for p in d.iterdir()if p.is_file()})
        (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'full-inputs.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=2)as tar:
        tar.add(dest,arcname='full_data');tar.add(ROOT/'data/columns.json',arcname='data/columns.json')
    (ROOT/'STAGING.json').write_text(json.dumps(dict(status='complete',archive_sha256=sha(archive),
        archive_bytes=archive.stat().st_size,folds=5,outer_validation_labels_not_exported=True,
        default_target_values_in_each_context_still_require_fold_local_training_only_use=True),indent=2)+'\n')
    print(json.dumps(dict(status='complete',bytes=archive.stat().st_size)),flush=True)
if __name__=='__main__':main()
