# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 27 | 0.37037037037037035 | 0.4444444444444444 |
| 0.200000 | -3 | 49 | 0.1836734693877551 | 0.24489795918367346 |
| 0.323686 | -5 | 46 | 0.13043478260869565 | 0.2391304347826087 |
| 0.400000 | 0 | 58 | 0.1206896551724138 | 0.1206896551724138 |
| 0.500000 | 0 | 59 | 0.01694915254237288 | 0.01694915254237288 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.002408 | -0.000062 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001744 | -0.000136 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000554 | -0.000083 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000431 | -0.000012 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000430 | -0.000020 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000099 | +0.000075 |
| employment_type | self_employed | 20987 | 2895 | +0.000055 | -0.000013 |
| employment_type | salaried | 33473 | 4432 | +0.000030 | -0.000009 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000038 | -0.000014 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000047 | +0.000022 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000057 | +0.000037 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000093 | -0.000019 |
| employment_type | variable_income | 15540 | 2180 | -0.000126 | +0.000009 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000172 | -0.000026 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000223 | +0.000021 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000463 | -0.000004 |
| aux_account_months | zero | 21000 | 2461 | -0.000464 | +0.000055 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000476 | -0.000017 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000962 | +0.000006 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001062 | +0.000014 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001341 | -0.000113 |
| baseline_risk_band | under005 | 29853 | 535 | -0.002033 | +0.000116 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.