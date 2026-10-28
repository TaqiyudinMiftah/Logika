# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 128 | 0.390625 | 0.390625 |
| 0.200000 | 2 | 176 | 0.23295454545454544 | 0.2215909090909091 |
| 0.323686 | -8 | 216 | 0.125 | 0.16203703703703703 |
| 0.400000 | -1 | 235 | 0.09787234042553192 | 0.10212765957446808 |
| 0.500000 | 5 | 234 | 0.0641025641025641 | 0.042735042735042736 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002218 | -0.000551 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001844 | -0.000231 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000903 | +0.000111 |
| employment_type | salaried | 33473 | 4432 | +0.000835 | -0.000134 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000369 | -0.000136 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000059 | -0.000145 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000058 | -0.000199 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000034 | -0.000230 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000213 | -0.000757 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000249 | +0.000037 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000293 | -0.001171 |
| aux_account_months | zero | 21000 | 2461 | -0.000928 | -0.000122 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001071 | -0.000218 |
| employment_type | variable_income | 15540 | 2180 | -0.001211 | -0.000046 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001276 | -0.001138 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001359 | -0.000169 |
| employment_type | self_employed | 20987 | 2895 | -0.001381 | -0.000234 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001723 | -0.000287 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001824 | -0.000092 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001848 | -0.000146 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001880 | -0.000104 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003496 | -0.000006 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.