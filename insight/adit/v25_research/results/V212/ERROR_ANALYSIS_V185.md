# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 34 | 0.4117647058823529 | 0.38235294117647056 |
| 0.200000 | -7 | 58 | 0.20689655172413793 | 0.3275862068965517 |
| 0.323686 | 2 | 72 | 0.20833333333333334 | 0.18055555555555555 |
| 0.400000 | 2 | 70 | 0.07142857142857142 | 0.04285714285714286 |
| 0.500000 | 1 | 75 | 0.09333333333333334 | 0.08 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.001875 | -0.000291 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001707 | +0.000065 |
| employment_type | self_employed | 20987 | 2895 | +0.001561 | +0.000055 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001357 | +0.000048 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000843 | +0.000219 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000450 | -0.000008 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000379 | +0.000023 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000366 | -0.000034 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000277 | +0.000001 |
| employment_type | salaried | 33473 | 4432 | +0.000088 | -0.000029 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000040 | +0.000006 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000070 | -0.000007 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000084 | -0.000006 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000320 | +0.000275 |
| aux_account_months | zero | 21000 | 2461 | -0.000377 | +0.000037 |
| employment_type | variable_income | 15540 | 2180 | -0.000668 | +0.000003 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000891 | -0.000212 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001148 | +0.000083 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.