# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 55 | 0.32727272727272727 | 0.36363636363636365 |
| 0.200000 | 2 | 86 | 0.2558139534883721 | 0.23255813953488372 |
| 0.323686 | -5 | 96 | 0.07291666666666667 | 0.125 |
| 0.400000 | 0 | 112 | 0.09821428571428571 | 0.09821428571428571 |
| 0.500000 | 1 | 101 | 0.04950495049504951 | 0.039603960396039604 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.003618 | -0.000202 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.001647 | -0.000008 |
| employment_type | salaried | 33473 | 4432 | +0.000844 | -0.000074 |
| employment_type | variable_income | 15540 | 2180 | +0.000791 | -0.000042 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000232 | -0.000514 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000214 | -0.000056 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000208 | -0.000356 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000161 | -0.000096 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000237 | -0.000108 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000286 | -0.000116 |
| aux_account_months | zero | 21000 | 2461 | -0.000352 | -0.000054 |
| employment_type | self_employed | 20987 | 2895 | -0.000415 | -0.000061 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000450 | -0.000028 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000514 | -0.000065 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000586 | -0.000446 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000725 | -0.000068 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000800 | -0.000168 |
| orders90_exposure_band | zero | 3428 | 106 | -0.000903 | +0.000179 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000972 | -0.000048 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001184 | -0.000121 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001980 | +0.000023 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002089 | -0.000462 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.