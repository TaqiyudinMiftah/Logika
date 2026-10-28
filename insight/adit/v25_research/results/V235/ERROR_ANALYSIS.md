# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 26 | 0.46153846153846156 | 0.38461538461538464 |
| 0.200000 | 0 | 41 | 0.24390243902439024 | 0.24390243902439024 |
| 0.323686 | 0 | 47 | 0.14893617021276595 | 0.14893617021276595 |
| 0.400000 | -2 | 59 | 0.0847457627118644 | 0.11864406779661017 |
| 0.500000 | 0 | 51 | 0.0196078431372549 | 0.0196078431372549 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.006020 | +0.000014 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.001466 | -0.000084 |
| employment_type | variable_income | 15540 | 2180 | +0.000753 | +0.000001 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.000327 | +0.000020 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000321 | -0.000001 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000291 | +0.000007 |
| employment_type | salaried | 33473 | 4432 | +0.000283 | +0.000005 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000237 | -0.000005 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000236 | +0.000190 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000189 | -0.000113 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000171 | +0.000028 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000020 | -0.000094 |
| employment_type | self_employed | 20987 | 2895 | -0.000083 | +0.000018 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000453 | +0.000016 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000456 | +0.000057 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000674 | +0.000128 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000754 | +0.000014 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000774 | -0.000059 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000842 | -0.000021 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000872 | +0.000020 |
| aux_account_months | zero | 21000 | 2461 | -0.000881 | +0.000037 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000931 | +0.000050 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.