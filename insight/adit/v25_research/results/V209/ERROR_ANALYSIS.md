# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -13 | 224 | 0.32589285714285715 | 0.38392857142857145 |
| 0.200000 | -4 | 322 | 0.21428571428571427 | 0.2267080745341615 |
| 0.323686 | -17 | 387 | 0.09302325581395349 | 0.13695090439276486 |
| 0.400000 | 1 | 435 | 0.09425287356321839 | 0.09195402298850575 |
| 0.500000 | -6 | 406 | 0.0665024630541872 | 0.0812807881773399 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.005232 | +0.000117 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.004572 | +0.000541 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.003021 | -0.000036 |
| employment_type | salaried | 33473 | 4432 | +0.002683 | +0.000023 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.002339 | -0.000047 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001894 | -0.000028 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001658 | +0.000859 |
| employment_type | self_employed | 20987 | 2895 | +0.001328 | -0.000030 |
| employment_type | variable_income | 15540 | 2180 | +0.001075 | +0.000326 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000837 | -0.000024 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000312 | +0.000058 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000254 | +0.000011 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000097 | +0.000049 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000119 | +0.000357 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000645 | +0.000219 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001032 | -0.000027 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001925 | +0.000165 |
| aux_account_months | zero | 21000 | 2461 | -0.002072 | +0.000331 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002637 | +0.000175 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002882 | +0.000110 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003174 | -0.000008 |
| orders90_exposure_band | zero | 3428 | 106 | -0.006918 | -0.000199 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.