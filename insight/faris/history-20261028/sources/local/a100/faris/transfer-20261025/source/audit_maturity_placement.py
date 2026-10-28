"""Audit new Kumo feature placement against explicit context-only prior identities."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import pandas as pd
import joblib
import baseline as b
from robustness import R,A,MAIN,PACKETS,read

parser=argparse.ArgumentParser()
parser.add_argument('--require-complete',action='store_true')
parser.add_argument('--hypothesis',choices=['H006','H007'],default='H006')
args=parser.parse_args()
directory,output=('core_maturity','CORE_MATURITY') if args.hypothesis=='H006' else ('core_late30','CORE_LATE30')
plan=read(R/f'{args.hypothesis}.json')
checks=[]
for f in range(5):
    source=A/f'repayment-severity-fe-20261021/fold{f}'
    transformer=read(source/'transformer.json')
    roles=read(MAIN/f'conditions/fold{f}/control90/boundary.json')
    assert transformer['fit_ids_sha256']==roles['context_ids_sha']
    assert transformer['fit_rows']==50000
    assert transformer['feature_fit_default_labels_opened']is False
    assert transformer['query_covariates_fit_shared_prior']is False
    assert b.sha(source/'features.parquet')==transformer['feature_hash']
    assert all(v['gradient_max_abs']<1e-4 for v in transformer['beta'].values())
    row=dict(fold=f,prior_fit_rows=50000,prior_context_ID_hash_matches=True,
             default_labels_used_in_prior=False,query_covariates_used_in_prior=False,
             prior_feature_table_sha256=transformer['feature_hash'],
             prior_parameters_sha256=b.sha(source/'transformer.json'))
    root=R/directory/f'fold{f}'
    if(root/'complete.json').exists():
        record=read(root/'complete.json')
        for name,digest in record['files'].items():assert b.sha(root/name)==digest
        saved=joblib.load(root/'canonical_context.joblib')
        original=pd.read_parquet(PACKETS/f'fold{f}/context.parquet')
        assert saved['context'].application_id.tolist()==original.application_id.tolist()
        assert np.array_equal(saved['context'].target.to_numpy(),original.target.to_numpy())
        assert saved['replacement']==plan['replacement']
        mature=pd.read_parquet(source/'features.parquet').set_index('application_id')
        for c in saved['columns']:
            if c not in saved['replacement']:
                pd.testing.assert_series_equal(saved['context'][c],original[c],check_exact=True)
            else:
                assert np.array_equal(saved['context'][c].to_numpy(),mature.loc[original.application_id,saved['replacement'][c]].to_numpy())
        row.update(canonical_context_verified=True,
                   core_input_columns_replaced=len(saved['replacement']),replacement_values_exact=True,
                   labels_and_IDs_unchanged=True,complete_source_sha256=b.sha(root/'complete.json'))
    checks.append(row)
completed=sum(row.get('canonical_context_verified',False) for row in checks)
if args.require_complete:assert completed==5
assert b.sha(A/'ranking-pipeline-20261020/assets/classifier.pt')==plan['checkpoint_sha256']
b.dump(dict(status='passed',scope='Context-only maturity priors and completed new Kumo contexts. No new default model fits or scores.',checks=checks,
            completed_contexts=completed,all_five_contexts_verified=(completed==5),checkpoint_hash_verified=True,
            inherited_snapshot_limitation_retained=True,independent_generalization_certified=False,
            audit_source_sha256=b.sha(Path(__file__))),R/f'{output}_BOUNDARY_AUDIT.json')
