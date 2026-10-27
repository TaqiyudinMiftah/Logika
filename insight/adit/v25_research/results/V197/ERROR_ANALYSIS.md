# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -7 | 226 | 0.3141592920353982 | 0.34513274336283184 |
| 0.200000 | -4 | 329 | 0.21580547112462006 | 0.22796352583586627 |
| 0.323686 | -6 | 379 | 0.1266490765171504 | 0.1424802110817942 |
| 0.400000 | 3 | 411 | 0.10462287104622871 | 0.09732360097323602 |
| 0.500000 | -2 | 390 | 0.07435897435897436 | 0.07948717948717948 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.009271 | +0.000543 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.004893 | -0.000015 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.002507 | -0.000494 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002092 | +0.000402 |
| employment_type | variable_income | 15540 | 2180 | +0.002090 | +0.000291 |
| employment_type | salaried | 33473 | 4432 | +0.001477 | -0.000071 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.001353 | -0.000164 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001303 | +0.000694 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000524 | -0.000151 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000382 | -0.001604 |
| employment_type | self_employed | 20987 | 2895 | -0.000123 | -0.000105 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000464 | +0.000007 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000497 | -0.000040 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000580 | +0.000151 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000724 | -0.000027 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001427 | +0.000100 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001429 | +0.000016 |
| aux_account_months | zero | 21000 | 2461 | -0.001712 | +0.000108 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002253 | -0.000179 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002357 | -0.000040 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002461 | -0.000005 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002709 | -0.000051 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.