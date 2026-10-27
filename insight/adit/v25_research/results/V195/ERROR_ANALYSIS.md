# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -4 | 178 | 0.29775280898876405 | 0.3202247191011236 |
| 0.200000 | -1 | 260 | 0.2230769230769231 | 0.22692307692307692 |
| 0.323686 | -15 | 296 | 0.0945945945945946 | 0.14527027027027026 |
| 0.400000 | -6 | 342 | 0.08771929824561403 | 0.10526315789473684 |
| 0.500000 | -1 | 337 | 0.06231454005934718 | 0.06528189910979229 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.006123 | +0.000228 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002188 | -0.000046 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.002085 | +0.000056 |
| employment_type | salaried | 33473 | 4432 | +0.001788 | +0.000081 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.001638 | +0.000170 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000832 | +0.001313 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000790 | +0.000053 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000551 | +0.000077 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000451 | +0.000281 |
| employment_type | variable_income | 15540 | 2180 | +0.000365 | +0.000107 |
| aux_account_months | zero | 21000 | 2461 | +0.000320 | +0.000224 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000197 | +0.000028 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000260 | +0.000219 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000409 | +0.000120 |
| employment_type | self_employed | 20987 | 2895 | -0.000481 | +0.000156 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000771 | +0.000542 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000934 | +0.000057 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001021 | +0.000155 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001322 | -0.000038 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001648 | +0.000184 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002304 | +0.000157 |
| orders90_exposure_band | zero | 3428 | 106 | -0.007827 | -0.000045 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.