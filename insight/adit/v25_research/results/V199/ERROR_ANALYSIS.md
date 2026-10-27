# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 117 | 0.3418803418803419 | 0.28205128205128205 |
| 0.200000 | 12 | 194 | 0.25257731958762886 | 0.19072164948453607 |
| 0.323686 | -10 | 239 | 0.08786610878661087 | 0.1297071129707113 |
| 0.400000 | 0 | 270 | 0.09259259259259259 | 0.09259259259259259 |
| 0.500000 | -7 | 270 | 0.04814814814814815 | 0.07407407407407407 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.003370 | +0.000415 |
| employment_type | variable_income | 15540 | 2180 | +0.003298 | +0.000056 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002882 | -0.000985 |
| baseline_risk_band | under005 | 29853 | 535 | +0.002094 | +0.001290 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.001550 | +0.000108 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.001053 | +0.000084 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001038 | +0.000206 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000833 | +0.000214 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000689 | +0.000152 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000642 | +0.000122 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000038 | +0.000184 |
| employment_type | salaried | 33473 | 4432 | -0.000188 | +0.000191 |
| employment_type | self_employed | 20987 | 2895 | -0.000304 | -0.000052 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000781 | +0.000041 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001227 | +0.000629 |
| aux_account_months | zero | 21000 | 2461 | -0.001281 | -0.000075 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001423 | +0.000353 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001687 | -0.000032 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002040 | +0.000019 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002113 | -0.000097 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002181 | -0.000082 |
| orders90_exposure_band | zero | 3428 | 106 | -0.007849 | -0.000701 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.