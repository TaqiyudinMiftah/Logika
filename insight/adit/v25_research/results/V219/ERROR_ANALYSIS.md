# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 293 | 0.3720136518771331 | 0.378839590443686 |
| 0.200000 | 25 | 435 | 0.24597701149425288 | 0.18850574712643678 |
| 0.323686 | -16 | 569 | 0.11072056239015818 | 0.13884007029876977 |
| 0.400000 | -4 | 616 | 0.08441558441558442 | 0.09090909090909091 |
| 0.500000 | -3 | 587 | 0.06473594548551959 | 0.06984667802385008 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.008445 | +0.001255 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.007944 | -0.002329 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.007455 | -0.000640 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.002616 | +0.001334 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002101 | -0.000325 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001746 | -0.001245 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.001016 | +0.000073 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000882 | +0.000169 |
| employment_type | salaried | 33473 | 4432 | +0.000810 | +0.000150 |
| employment_type | variable_income | 15540 | 2180 | +0.000225 | -0.000080 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000063 | +0.000197 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000482 | +0.000139 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000722 | +0.000004 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000777 | +0.000385 |
| employment_type | self_employed | 20987 | 2895 | -0.001435 | +0.000118 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001783 | -0.000767 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002569 | -0.000087 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002893 | +0.000138 |
| aux_account_months | zero | 21000 | 2461 | -0.003003 | +0.000320 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003638 | -0.000031 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003729 | +0.000159 |
| orders90_exposure_band | zero | 3428 | 106 | -0.011462 | +0.001238 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.