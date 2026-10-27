# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 32 | 0.53125 | 0.40625 |
| 0.200000 | 4 | 61 | 0.26229508196721313 | 0.19672131147540983 |
| 0.323686 | -5 | 66 | 0.07575757575757576 | 0.15151515151515152 |
| 0.400000 | -4 | 78 | 0.07692307692307693 | 0.1282051282051282 |
| 0.500000 | 1 | 83 | 0.03614457831325301 | 0.024096385542168676 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.001617 | +0.000531 |
| employment_type | variable_income | 15540 | 2180 | +0.000922 | +0.000051 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000610 | +0.000102 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000607 | +0.000028 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000602 | +0.000043 |
| employment_type | salaried | 33473 | 4432 | +0.000574 | +0.000078 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000165 | +0.000507 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000113 | +0.000137 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000041 | +0.000187 |
| employment_type | self_employed | 20987 | 2895 | -0.000055 | +0.000036 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000074 | +0.000028 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000123 | +0.000120 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000208 | +0.000112 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000263 | +0.000080 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000304 | +0.000063 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000314 | +0.000022 |
| aux_account_months | zero | 21000 | 2461 | -0.000327 | +0.000161 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000373 | -0.000062 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000463 | +0.000045 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000702 | -0.000157 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000843 | +0.000219 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001268 | +0.000033 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.