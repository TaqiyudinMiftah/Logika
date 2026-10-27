# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -7 | 246 | 0.33739837398373984 | 0.36585365853658536 |
| 0.200000 | 4 | 359 | 0.233983286908078 | 0.22284122562674094 |
| 0.323686 | -17 | 426 | 0.10328638497652583 | 0.1431924882629108 |
| 0.400000 | -4 | 460 | 0.08260869565217391 | 0.09130434782608696 |
| 0.500000 | -2 | 446 | 0.07174887892376682 | 0.07623318385650224 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.006123 | +0.000228 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002747 | +0.000887 |
| employment_type | salaried | 33473 | 4432 | +0.002561 | +0.000045 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002188 | -0.000046 |
| employment_type | variable_income | 15540 | 2180 | +0.001123 | +0.000334 |
| employment_type | self_employed | 20987 | 2895 | +0.000886 | -0.000048 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000551 | +0.000077 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000314 | +0.000898 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000294 | -0.000009 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000197 | +0.000028 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000006 | -0.000313 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000418 | +0.000100 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000503 | +0.000064 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001322 | -0.000038 |
| aux_account_months | zero | 21000 | 2461 | -0.001845 | +0.000310 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002304 | +0.000157 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003383 | +0.000081 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003403 | +0.000132 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.