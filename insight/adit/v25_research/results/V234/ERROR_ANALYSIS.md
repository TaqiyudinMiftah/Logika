# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 25 | 0.36 | 0.4 |
| 0.200000 | 2 | 44 | 0.22727272727272727 | 0.18181818181818182 |
| 0.323686 | -4 | 49 | 0.12244897959183673 | 0.20408163265306123 |
| 0.400000 | 3 | 55 | 0.14545454545454545 | 0.09090909090909091 |
| 0.500000 | -2 | 53 | 0.018867924528301886 | 0.05660377358490566 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001080 | -0.000040 |
| employment_type | self_employed | 20987 | 2895 | +0.000829 | -0.000042 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000430 | +0.000051 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000394 | -0.000046 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000366 | -0.000062 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000223 | -0.000023 |
| aux_account_months | zero | 21000 | 2461 | +0.000176 | +0.000040 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000144 | +0.000016 |
| employment_type | variable_income | 15540 | 2180 | +0.000126 | +0.000005 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000076 | +0.000037 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000021 | -0.000001 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000146 | -0.000037 |
| employment_type | salaried | 33473 | 4432 | -0.000214 | -0.000026 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000231 | -0.000005 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000258 | -0.000037 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000350 | +0.000002 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000361 | -0.000310 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000602 | -0.000064 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000778 | -0.000046 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000811 | -0.000121 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001433 | +0.000250 |
| orders90_exposure_band | zero | 3428 | 106 | -0.003010 | -0.000108 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.