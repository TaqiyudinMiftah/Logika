# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 3 | 25 | 0.44 | 0.32 |
| 0.200000 | 3 | 41 | 0.24390243902439024 | 0.17073170731707318 |
| 0.323686 | -6 | 39 | 0.0 | 0.15384615384615385 |
| 0.400000 | -2 | 54 | 0.09259259259259259 | 0.12962962962962962 |
| 0.500000 | 3 | 55 | 0.07272727272727272 | 0.01818181818181818 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.000644 | +0.000395 |
| employment_type | variable_income | 15540 | 2180 | +0.000453 | +0.000051 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000424 | +0.000077 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000310 | +0.000017 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000254 | +0.000332 |
| employment_type | self_employed | 20987 | 2895 | +0.000152 | +0.000019 |
| employment_type | salaried | 33473 | 4432 | +0.000105 | +0.000074 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000035 | +0.000027 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000012 | +0.000029 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000102 | +0.000024 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000134 | +0.000284 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000134 | +0.000121 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000150 | +0.000304 |
| aux_account_months | zero | 21000 | 2461 | -0.000176 | +0.000123 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000195 | +0.000028 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000325 | +0.000098 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000337 | +0.000018 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000357 | +0.000093 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000360 | +0.000013 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000431 | +0.000082 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000474 | -0.000076 |
| orders90_exposure_band | zero | 3428 | 106 | -0.001505 | +0.000031 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.