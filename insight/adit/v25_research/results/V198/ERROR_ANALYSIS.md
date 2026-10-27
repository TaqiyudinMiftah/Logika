# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -3 | 238 | 0.3403361344537815 | 0.35294117647058826 |
| 0.200000 | 23 | 363 | 0.24793388429752067 | 0.18457300275482094 |
| 0.323686 | -12 | 449 | 0.12249443207126949 | 0.1492204899777283 |
| 0.400000 | 7 | 494 | 0.09716599190283401 | 0.08299595141700405 |
| 0.500000 | -11 | 484 | 0.05371900826446281 | 0.07644628099173553 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.010601 | +0.002966 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.009498 | -0.000161 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.006166 | +0.000509 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.003455 | +0.001422 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.001073 | +0.000230 |
| employment_type | salaried | 33473 | 4432 | +0.000989 | +0.000388 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000214 | -0.000526 |
| employment_type | variable_income | 15540 | 2180 | +0.000112 | +0.000112 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000030 | +0.000658 |
| aux_account_months | zero | 21000 | 2461 | -0.000054 | +0.000474 |
| employment_type | self_employed | 20987 | 2895 | -0.000595 | +0.000067 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000955 | +0.000161 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001043 | +0.000110 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001189 | +0.000267 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001301 | +0.000193 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001391 | +0.000301 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001713 | +0.000147 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001792 | +0.000204 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001889 | +0.000478 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001978 | +0.000315 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002510 | +0.001309 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002541 | +0.000228 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.