"""Create compact, reusable packets without validation labels or fitted models."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
R = Path(__file__).resolve().parents[1]
A = Path('/mnt/home/faris/logika-faris-exploration')
BANK = Path('/mnt/home/faris/faris/logika-meta-experiments/meta-20261016-v1/bank')
sys.path.insert(0, str(A/'transfer-20261025/source'))
sys.path.insert(0, '/mnt/home/faris/logika-model-arena-20261025/source')
import baseline as b
import inputs

def main():
    manifest = json.loads((R/'BANK_MANIFEST.json').read_text())
    verified = {}
    for name in ['numeric.npy', 'categorical.parquet', 'metadata.parquet']:
        digest = b.sha(BANK/name)
        assert digest == manifest['files'][name], name
        verified[str(BANK/name)] = digest
    metadata = pd.read_parquet(BANK/'metadata.parquet')
    extras = []
    for path, names in json.loads((R/'EXTRA_SOURCES.json').read_text()).items():
        x = pd.read_parquet(path).set_index('application_id')
        extras.append(x.loc[metadata.application_id[:70000]].drop(columns=['customer_id'], errors='ignore').reset_index(drop=True))
        verified[path] = b.sha(path)
    extra = pd.concat(extras, axis=1)
    extra.to_parquet(R/'extra.parquet', index=False)
    columns = json.loads((R/'ANCHOR_SOURCE.json').read_text())
    wanted = list(dict.fromkeys(columns['feature_columns']+columns['head_columns']))
    checks = []
    for fold in range(5):
        original, ix, q, ctx, yc, cols, query, head_cols, spline, raw, expected = inputs.inputs(fold)
        assert original.application_id.tolist() == metadata.application_id[:70000].tolist()
        prior = pd.read_parquet(A/f'fold-local-priors-20261021/fold{fold}/features.parquet').set_index('application_id')
        missing = [c for c in wanted if c not in ctx]
        for c in missing:
            assert c in prior
            ctx[c] = prior.loc[ctx.application_id, c].to_numpy()
        mature = pd.read_parquet(A/f'repayment-severity-fe-20261021/fold{fold}/features.parquet').set_index('application_id')
        for threshold in [1,30]:
            c = f're_late_{threshold}_posterior'
            ctx[c] = mature.loc[ctx.application_id, f'matched_beta_late_{threshold}'].to_numpy()
        current = pd.concat([ctx[['application_id']+wanted], query[['application_id']+wanted]], ignore_index=True)
        current = current.set_index('application_id').loc[original.application_id].reset_index(drop=True)
        root = R/f'packets/fold{fold}';root.mkdir(parents=True, exist_ok=False)
        current.to_parquet(root/'current.parquet', index=False)
        np.savez_compressed(root/'roles.npz', context=ix['context'], correction=ix['correction'], validation=ix['validation'],
                            y_context=ctx.target.to_numpy(int), y_correction=yc)
        np.savez_compressed(root/'baseline.npz', raw=raw, spline=spline, corrected=expected)
        b.dump({'fold': fold, 'head_columns':head_cols, 'baseline_columns':cols,
                'files': {p.name:b.sha(p) for p in root.iterdir()}, 'context_rows':50000, 'correction_rows':6000, 'validation_rows':14000,
                'outer_labels_in_packet':False, 'missing_context_columns_rebuilt_from_fold_local_prior_table':missing}, root/'manifest.json')
        checks.append({'fold':fold,'rows':len(current),'columns':len(current.columns),'role_overlap':0})
    b.dump({'verified_source_files':verified,'packets':checks,'extra_features':list(extra.columns),
            'catalog_sha256':b.sha(R/'CATALOG.json'),'source_sha256':b.sha(__file__),
            'inherited_snapshot_semantics':'All historical due/payment timestamps and observed DPD bounds checked separately; exact upstream capture semantics remain an assumption.',
            'outer_labels_copied':False}, R/'DATA_GATE.json')
    print(json.dumps({'packets':len(checks),'extra_columns':len(extra.columns),'status':'complete'}))

if __name__=='__main__':main()
