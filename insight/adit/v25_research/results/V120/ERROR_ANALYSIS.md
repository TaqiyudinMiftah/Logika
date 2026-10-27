# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 161 | 0.37888198757763975 | 0.3167701863354037 |
| 0.200000 | 13 | 258 | 0.24031007751937986 | 0.18992248062015504 |
| 0.323686 | -20 | 315 | 0.09523809523809523 | 0.15873015873015872 |
| 0.400000 | -4 | 362 | 0.0856353591160221 | 0.09668508287292818 |
| 0.500000 | 1 | 363 | 0.06887052341597796 | 0.06611570247933884 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.010218 | +0.002220 |
| baseline_risk_band | under005 | 29853 | 535 | +0.008623 | +0.002239 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.003323 | +0.001756 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001555 | -0.000258 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001219 | +0.000612 |
| employment_type | variable_income | 15540 | 2180 | +0.001208 | +0.000353 |
| employment_type | self_employed | 20987 | 2895 | +0.000031 | +0.000219 |
| aux_account_months | zero | 21000 | 2461 | -0.000033 | +0.000379 |
| employment_type | salaried | 33473 | 4432 | -0.000067 | +0.000403 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000694 | +0.000216 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000761 | +0.000474 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000801 | +0.000065 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001015 | +0.000370 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001058 | +0.000529 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001232 | +0.000406 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002104 | +0.000323 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002427 | +0.000943 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002496 | +0.000308 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.