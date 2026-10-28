# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 114 | 0.43859649122807015 | 0.40350877192982454 |
| 0.200000 | -13 | 203 | 0.19704433497536947 | 0.26108374384236455 |
| 0.323686 | -7 | 282 | 0.1099290780141844 | 0.1347517730496454 |
| 0.400000 | -6 | 303 | 0.07260726072607261 | 0.0924092409240924 |
| 0.500000 | -9 | 320 | 0.046875 | 0.075 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.013745 | +0.001318 |
| baseline_risk_band | under005 | 29853 | 535 | +0.004687 | +0.002806 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.002025 | -0.000165 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001421 | -0.000034 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001391 | -0.000001 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.001295 | -0.000145 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.001246 | +0.000036 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000843 | +0.000072 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000731 | +0.000537 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000595 | +0.000136 |
| employment_type | variable_income | 15540 | 2180 | +0.000509 | -0.000154 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000442 | +0.000010 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000384 | +0.000127 |
| aux_account_months | zero | 21000 | 2461 | +0.000338 | +0.000055 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000136 | +0.000092 |
| employment_type | salaried | 33473 | 4432 | -0.000199 | +0.000169 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000987 | +0.000071 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001061 | +0.000102 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001422 | -0.000213 |
| employment_type | self_employed | 20987 | 2895 | -0.001562 | +0.000078 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001710 | -0.000029 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001721 | -0.000150 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.