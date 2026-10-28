# Pre-decision feature experiments — 25 October 2026

All three prespecified families lowered original-five-fold pooled KS on both cached Kumo seeds. The monthly alignment mechanism did not outperform its within-borrower calendar permutation. No model was promoted or submitted.

| Addition to mature correction + spline | Original seed KS | Fresh seed KS | Mean KS | Mean delta vs B |
|---|---:|---:|---:|---:|
| liquidity | 0.528860488 | 0.528655885 | 0.528565802 | -0.001502719 |
| alignment | 0.527683784 | 0.527115575 | 0.527344923 | -0.002723598 |
| application_path | 0.528223534 | 0.528429143 | 0.528457675 | -0.001610846 |
| alignment_null | 0.527780493 | 0.527500707 | 0.527490176 | -0.002578345 |

Each condition retains 5 spline fits and 10 correction heads (60 serialized estimators across four conditions). Kumo raw predictions are unchanged. All ten control seed/fold replays passed to less than 2e-14 maximum absolute error.

The 24 fields were fixed before development-label residual diagnostics. New records use complete account months and shopping/prior-application events before the selected decision; no new repayment snapshot values were read. Availability is balanced between train and test: 70% of borrowers have the principal account/history fields. Missingness for conditional cancellation/card metrics is much higher and is retained explicitly.

These are reused-development comparisons, not independent confirmation. The inherited baseline still has unresolved repayment-snapshot timing provenance. Frozen selection required a gain on both seeds, pooled KS above 0.531505932, and at least three of five fold wins. No candidate met these requirements.

Evidence: `FEATURE_PLAN.json`, `features/GATE.json`, `EDA.json`, `RESULT.json`, `UNCERTAINTY.json`, `COMPUTE_LEDGER.json`, and `matched/*/fold*/`.
