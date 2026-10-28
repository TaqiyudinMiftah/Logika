"""Verify the frozen core intervention in each full-data context; never fit."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import baseline as b
from robustness import R,A,read
from maturity_features import MaturityFeatures

full=A/'submission-full-periodicity-20261021'
dest=R/'candidate_D'
plan=read(dest/'FULL_REFIT_PLAN.json')
tag={'H006':'CORE_MATURITY','H007':'CORE_LATE30'}[plan['core_hypothesis']]
replacement_spec=read(R/f'{plan["core_hypothesis"]}.json')['replacement']
assert replacement_spec==plan['replacement']
assert read(R/f'{tag}_RESULT.json')['promotion_predicate_passed'] is True
assert b.sha(R/f'{tag}_RESULT.json')==plan['selection_evidence']['sha256']
apps=pd.read_parquet(full/'applications.parquet')
history=pd.read_csv(A/'exploration-12h-20261017/raw/repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'])
checks=[]
for f in range(5):
    root=dest/f'final/fold{f}'
    before=joblib.load(full/f'models/fold{f}/kumo_model.joblib')
    after=joblib.load(root/'kumo_model.joblib')
    roles=np.load(full/f'input/artifacts/fold{f}/indices.npz')
    context_ids=apps.application_id.iloc[roles['context']]
    transformer=MaturityFeatures.load(root/'maturity_transformer.json')
    parameters=transformer.parameters
    assert parameters['fit_rows']==64000
    assert parameters['fit_application_ids_sha256']==b.ids_hash(apps,roles['context'])
    assert parameters['default_labels_used'] is False
    assert b.sha(root/'maturity_transformer.json')==b.sha(R/f'candidate_B/final/fold{f}/maturity_transformer.json')
    assert after['columns']==before['columns'] and len(after['columns'])==90
    assert after['replacement']==replacement_spec
    assert after['seeds']==[2026+f,12026+f] and after['num_estimators']==8
    assert after['context'].application_id.tolist()==context_ids.tolist()
    mature=transformer.transform(apps,history).set_index('application_id')
    for c in after['context']:
        if c in replacement_spec:
            threshold=1 if replacement_spec[c].endswith('_1') else 30
            expected=mature.loc[context_ids,f're_late_{threshold}_posterior'].to_numpy()
            assert np.array_equal(after['context'][c].to_numpy(),expected)
        else:
            pd.testing.assert_series_equal(after['context'][c],before['context'][c],check_exact=True)
    for mode in ['original','fresh']:
        raw=np.load(root/f'raw_{mode}.npz')
        rows=np.r_[roles['correction'],np.arange(70000,100000)]
        assert np.array_equal(raw['rows'],rows)
        assert raw['application_id'].tolist()==apps.application_id.iloc[rows].tolist()
        assert np.isfinite(raw['prediction']).all()
    assert b.sha(root/'spline.joblib')==b.sha(R/f'candidate_B/final/fold{f}/spline.joblib')
    pd.testing.assert_frame_equal(pd.read_parquet(root/'test_features.parquet'),pd.read_parquet(R/f'candidate_B/final/fold{f}/test_features.parquet'),check_exact=True)
    checks.append(dict(fold=f,exactly_declared_core_columns_changed=True,core_columns_changed=len(replacement_spec),context_labels_unchanged=True,
                       context_only_prior_rows=64000,downstream_features_match_B=True,
                       spline_identical_to_B=True,raw_query_roles_verified=True))
b.dump(dict(status='passed',scope='Full-data core-input and provenance audit; no fitting or score selection',
            checks=checks,independent_generalization_certified=False,
            source_sha256=b.sha(Path(__file__))),dest/'CORE_FULL_BOUNDARY_AUDIT.json')
