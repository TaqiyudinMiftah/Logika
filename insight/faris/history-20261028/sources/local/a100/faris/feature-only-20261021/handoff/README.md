# Frozen-model repayment maturity features

This package reproduces the actual feature-only experiments. It does not tune or select a predictive model, optimize ensemble weights, generate a submission, or promise a public leaderboard gain.

## Current evidence

| Fixed pipeline | Pooled5foldKS | AUC |
|---|---:|---:|
| Repaired V25 baseline | .5302185367 | .8470258500 |
| >=30day posterior means in correction only | .5306046301 | .8470538256 |
| Same means in correction and existing spline | .5306361042 | .8470560913 |

The last is a provisional development lead. Its simultaneous conditional delta intervals include zero: KS[-.00079716,.00146547], AUC[-.00004696,.00009552]. These reused labels and conditional intervals do not establish independent generalization. Target.54 remains unmet. Terminal confirmation has not been opened during feature-only research.

The additional five eligible-denominator fields completed all5folds and were negative: KS.5289636640/AUC.8469409180. Their values also filled previously missing histories with priors. A separately frozen diagnostic now preserves those original missingness masks; it is not yet scored and is not a supported improvement. Negative cash-flow exposure, bureau trajectory, shopping change, joint severity and loan-cluster comparisons remain retained in sibling study directories. The persistent-provider-effect hypothesis was excluded without a prediction fit after its label-free diagnostic did not support it. The nondefault `eligible_denominators` mode remains solely for reproducing the negative evidence.

## Feature definitions

`posterior_means_only` replaces two existing fields, without adding model inputs:

* `re_late_1_posterior`: beta-binomial posterior mean of maxDPD>=1 among installments at least30days old at decision.
* `re_late_30_posterior`: the corresponding maxDPD>=30 posterior on the same event set.

Both Beta priors are fitted on the explicit50kcontext applicants only. This aligns opportunity maturity rather than treating recently due installments as comparable evidence. Historical maxDPD remains an as-supplied snapshot field with unresolved provenance; time filtering is not certification of online availability. No amounts, new DPD fields, IDs or order predictors are introduced. Existing posterior-standard-deviation fields remain unchanged in this isolated feature experiment.

`eligible_denominators` additionally replaces the five selected legacy v3 aliases: all-history thresholds1/7/30/90 and180day threshold1. For each threshold, numerator and denominator use only due events old enough to reach it; prevalence comes only from eligible context events. Shrinkage stays24. Zero eligible history maps to the context prior. Alias names and input order remain fixed; these documented formulas replace their old all-opportunity semantics.

All missing-value handling, spline knots/scales, correction coefficients and classifier fitting remain inside the original training partitions. The feature factory itself accepts no default labels. Customer and application IDs align histories; neither is returned as a predictor.

## Usage

```python
import pandas as pd
from maturity_features import MaturityFeatures

# Pass only label-free columns. Context IDs must be the frozen fold's50k IDs.
apps = pd.concat([
    pd.read_csv('train_applications.csv', usecols=[
        'application_id', 'customer_id', 'decision_datetime']),
    pd.read_csv('test_applications.csv', usecols=[
        'application_id', 'customer_id', 'decision_datetime']),
], ignore_index=True)
history = pd.read_csv('repayment_history.csv', usecols=[
    'customer_id', 'installment_id', 'due_date', 'max_days_past_due'])

factory = MaturityFeatures('posterior_means_only')
factory.fit(apps, history, frozen_context_application_ids)
features = factory.transform(apps, history)
factory.save('fold_feature_transformer.json')

# Replace the named fields in the existing feature frame after an ID join.
values = features.set_index('application_id').loc[existing_frame.application_id]
for column in features.columns.drop('application_id'):
    existing_frame[column] = values[column].to_numpy()
```

Do not fit the factory on the combined development/test population or on evaluation rows. The API requires explicit fitting IDs; validation uses the original50000/6000/14000context/correction/outer roles. One application per customer is a dataset invariant enforced by this implementation.

`fitted/posterior_means_only/fold0.json` throughfold4 contain the actual reproduced validation priors. `fitted/eligible_denominators/` also contains the five context prevalences. Loading these requires no refitting. The factory has no dependency on the remote filesystem; only the verification script uses the campaign's archive paths.

## Verification

`FACTORY_AUDIT.json` records parity against the actual100kapplicant feature outputs, all7fields in all5folds. The verification also checks saved-transformer replay, identical two-field outputs between modes, exclusion of default-label columns, invariance of fitted priors when all noncontext history outcomes are changed, future-event exclusion, row-order invariance and application-ID invariance. This proves implementation parity and tested information boundaries; it does not prove predictive generalization.

Reproduce in the pinned campaign environment with `python verify_factory.py`. Full predictor/model evidence remains under `/mnt/home/logika-faris-exploration/repayment-severity-fe-20261021/`, `/mnt/home/logika-faris-exploration/maturity-placement-fe-20261021/` and `/mnt/home/logika-faris-exploration/mature-denominator-fe-20261021/` onPC2. Model parameters, original folds/seeds, correction strength and75:25blend are unchanged. No final refit or submission is included.
