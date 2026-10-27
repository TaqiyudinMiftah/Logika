# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 28 | 0.35714285714285715 | 0.21428571428571427 |
| 0.200000 | 5 | 33 | 0.36363636363636365 | 0.21212121212121213 |
| 0.323686 | 0 | 32 | 0.15625 | 0.15625 |
| 0.400000 | 5 | 42 | 0.19047619047619047 | 0.07142857142857142 |
| 0.500000 | -1 | 37 | 0.02702702702702703 | 0.05405405405405406 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001782 | +0.000119 |
| aux_account_months | zero | 21000 | 2461 | +0.000482 | +0.000057 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000450 | +0.000037 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000438 | +0.000008 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000366 | +0.000065 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000361 | +0.000034 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000286 | +0.000033 |
| employment_type | salaried | 33473 | 4432 | +0.000284 | +0.000015 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000270 | -0.000009 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000243 | +0.000016 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000096 | +0.000003 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000075 | +0.000142 |
| employment_type | self_employed | 20987 | 2895 | +0.000014 | +0.000003 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000000 | +0.000000 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000006 | +0.000186 |
| employment_type | variable_income | 15540 | 2180 | -0.000153 | +0.000031 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000181 | +0.000008 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000208 | +0.000010 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000286 | +0.000025 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000415 | -0.000002 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000423 | +0.000034 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000662 | +0.000049 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.