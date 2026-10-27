# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 188 | 0.42021276595744683 | 0.3829787234042553 |
| 0.200000 | -10 | 285 | 0.17543859649122806 | 0.21052631578947367 |
| 0.323686 | -12 | 332 | 0.1144578313253012 | 0.15060240963855423 |
| 0.400000 | 7 | 377 | 0.11671087533156499 | 0.09814323607427056 |
| 0.500000 | -6 | 355 | 0.0647887323943662 | 0.08169014084507042 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.005564 | -0.000105 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001317 | +0.000889 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001244 | +0.000294 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.001126 | -0.000152 |
| employment_type | salaried | 33473 | 4432 | +0.000635 | +0.000026 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000377 | -0.000053 |
| employment_type | variable_income | 15540 | 2180 | -0.000547 | -0.000104 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000681 | +0.000025 |
| aux_account_months | zero | 21000 | 2461 | -0.000726 | +0.000027 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000941 | -0.000152 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001213 | +0.000081 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001287 | +0.000016 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001327 | -0.000321 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001652 | -0.000034 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001860 | +0.000032 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001870 | -0.000038 |
| employment_type | self_employed | 20987 | 2895 | -0.002969 | +0.000040 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003605 | -0.000115 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.