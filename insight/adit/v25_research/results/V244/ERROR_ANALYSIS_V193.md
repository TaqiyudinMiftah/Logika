# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 263 | 0.3726235741444867 | 0.33460076045627374 |
| 0.200000 | -2 | 404 | 0.2301980198019802 | 0.23514851485148514 |
| 0.323686 | 1 | 477 | 0.11949685534591195 | 0.11740041928721175 |
| 0.400000 | 5 | 505 | 0.09504950495049505 | 0.08514851485148515 |
| 0.500000 | -1 | 493 | 0.07099391480730223 | 0.07302231237322515 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6096 | 3493 | +0.004678 | +0.000135 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002837 | -0.000155 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002643 | -0.000152 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001814 | -0.000156 |
| aux_account_months | zero | 21000 | 2461 | +0.000346 | -0.000298 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.000409 | -0.001749 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000635 | -0.000300 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000872 | -0.000621 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001012 | -0.000160 |
| baseline_risk_band | under005 | 29938 | 544 | -0.001558 | -0.001002 |
| employment_type | variable_income | 15540 | 2180 | -0.001792 | -0.000401 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.002051 | -0.000400 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002518 | -0.000228 |
| employment_type | self_employed | 20987 | 2895 | -0.002571 | -0.000145 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002591 | -0.000167 |
| employment_type | salaried | 33473 | 4432 | -0.002672 | -0.000163 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002734 | -0.000188 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002860 | -0.000063 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.