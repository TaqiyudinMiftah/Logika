# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 150 | 0.3333333333333333 | 0.34 |
| 0.200000 | -5 | 219 | 0.1917808219178082 | 0.2146118721461187 |
| 0.323686 | -9 | 270 | 0.1037037037037037 | 0.13703703703703704 |
| 0.400000 | 4 | 316 | 0.11075949367088607 | 0.0981012658227848 |
| 0.500000 | -3 | 285 | 0.06666666666666667 | 0.07719298245614035 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.006636 | +0.000442 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.004805 | +0.000891 |
| baseline_risk_band | under005 | 29853 | 535 | +0.003854 | +0.000176 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.002335 | +0.000805 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.002209 | +0.000264 |
| employment_type | salaried | 33473 | 4432 | +0.001153 | +0.000148 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.000539 | +0.000310 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000458 | +0.000032 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000381 | +0.000106 |
| employment_type | variable_income | 15540 | 2180 | +0.000338 | +0.000213 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000038 | +0.000107 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000116 | +0.000091 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000166 | +0.000101 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000375 | +0.000117 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000489 | +0.000185 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000852 | +0.000101 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001249 | +0.000100 |
| employment_type | self_employed | 20987 | 2895 | -0.001353 | +0.000017 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002096 | +0.000412 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002244 | +0.000126 |
| aux_account_months | zero | 21000 | 2461 | -0.002370 | +0.000223 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002449 | +0.000149 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.