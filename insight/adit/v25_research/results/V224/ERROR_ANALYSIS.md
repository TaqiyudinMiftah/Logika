# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 34 | 0.4411764705882353 | 0.29411764705882354 |
| 0.200000 | -2 | 59 | 0.1864406779661017 | 0.22033898305084745 |
| 0.323686 | -5 | 61 | 0.09836065573770492 | 0.18032786885245902 |
| 0.400000 | 5 | 82 | 0.17073170731707318 | 0.10975609756097561 |
| 0.500000 | 4 | 89 | 0.06741573033707865 | 0.02247191011235955 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002408 | -0.000059 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001102 | +0.000339 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000609 | +0.000108 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000567 | +0.000150 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000522 | +0.000031 |
| employment_type | self_employed | 20987 | 2895 | +0.000442 | -0.000018 |
| aux_account_months | zero | 21000 | 2461 | +0.000435 | +0.000157 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000280 | +0.000009 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000210 | -0.000048 |
| employment_type | salaried | 33473 | 4432 | +0.000172 | +0.000051 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000075 | +0.000546 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000057 | +0.000076 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000142 | -0.000007 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000207 | +0.000015 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000212 | -0.000011 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000334 | +0.000104 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000337 | -0.000004 |
| employment_type | variable_income | 15540 | 2180 | -0.000417 | +0.000052 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000539 | +0.000067 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001078 | -0.000019 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001972 | +0.000178 |
| orders90_exposure_band | zero | 3428 | 106 | -0.003908 | -0.000043 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.