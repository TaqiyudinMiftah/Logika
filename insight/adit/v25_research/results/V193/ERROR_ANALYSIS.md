# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -8 | 245 | 0.3346938775510204 | 0.3673469387755102 |
| 0.200000 | 4 | 364 | 0.23626373626373626 | 0.22527472527472528 |
| 0.323686 | -12 | 425 | 0.10823529411764705 | 0.13647058823529412 |
| 0.400000 | -7 | 465 | 0.08172043010752689 | 0.0967741935483871 |
| 0.500000 | -2 | 442 | 0.07013574660633484 | 0.0746606334841629 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003962 | +0.000327 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.003632 | -0.000045 |
| employment_type | salaried | 33473 | 4432 | +0.002688 | +0.000040 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002242 | -0.000028 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001496 | +0.000840 |
| employment_type | variable_income | 15540 | 2180 | +0.001432 | +0.000340 |
| employment_type | self_employed | 20987 | 2895 | +0.001245 | -0.000047 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.001093 | +0.000077 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000817 | -0.000001 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000689 | +0.000029 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000024 | -0.000315 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000172 | +0.000055 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000180 | +0.000285 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000277 | +0.000123 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000375 | +0.000922 |
| aux_account_months | zero | 21000 | 2461 | -0.001403 | +0.000279 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001546 | -0.000023 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002528 | +0.000160 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002892 | +0.000110 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002998 | -0.000006 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003152 | +0.000065 |
| orders90_exposure_band | zero | 3428 | 106 | -0.007219 | -0.000270 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.