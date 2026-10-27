# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 32 | 0.40625 | 0.375 |
| 0.200000 | -3 | 49 | 0.2857142857142857 | 0.3469387755102041 |
| 0.323686 | -5 | 57 | 0.08771929824561403 | 0.17543859649122806 |
| 0.400000 | -1 | 51 | 0.0196078431372549 | 0.0392156862745098 |
| 0.500000 | 5 | 64 | 0.109375 | 0.03125 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.002315 | -0.000161 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000521 | -0.000041 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000471 | +0.000026 |
| employment_type | self_employed | 20987 | 2895 | +0.000401 | -0.000075 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000368 | -0.000012 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000343 | +0.000023 |
| aux_account_months | zero | 21000 | 2461 | +0.000324 | +0.000012 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000308 | +0.000243 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000261 | +0.000048 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000005 | -0.000014 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000135 | -0.000056 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000213 | -0.000007 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000246 | -0.000015 |
| employment_type | salaried | 33473 | 4432 | -0.000585 | +0.000015 |
| employment_type | variable_income | 15540 | 2180 | -0.000710 | +0.000042 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001098 | +0.000034 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001368 | -0.000114 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002654 | +0.000002 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.