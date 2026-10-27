# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 14 | 0.5 | 0.5 |
| 0.200000 | 3 | 24 | 0.20833333333333334 | 0.08333333333333333 |
| 0.323686 | -3 | 27 | 0.1111111111111111 | 0.2222222222222222 |
| 0.400000 | 5 | 19 | 0.3157894736842105 | 0.05263157894736842 |
| 0.500000 | 2 | 25 | 0.08 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.000484 | -0.000038 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000243 | -0.000202 |
| employment_type | self_employed | 20987 | 2895 | +0.000193 | -0.000028 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000054 | -0.000040 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000038 | -0.000034 |
| employment_type | salaried | 33473 | 4432 | +0.000015 | -0.000043 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000156 | -0.000035 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000181 | -0.000042 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000203 | -0.000041 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000231 | -0.000036 |
| aux_account_months | zero | 21000 | 2461 | -0.000284 | -0.000018 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000316 | -0.000025 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000478 | -0.000139 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000544 | -0.000067 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000565 | -0.000013 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001373 | -0.000220 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001612 | -0.000002 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.002066 | -0.000264 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.