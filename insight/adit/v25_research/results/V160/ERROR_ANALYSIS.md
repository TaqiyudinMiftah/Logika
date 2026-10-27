# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 50 | 0.3 | 0.3 |
| 0.200000 | 3 | 86 | 0.20930232558139536 | 0.1744186046511628 |
| 0.323686 | -5 | 101 | 0.10891089108910891 | 0.15841584158415842 |
| 0.400000 | -5 | 105 | 0.09523809523809523 | 0.14285714285714285 |
| 0.500000 | -3 | 121 | 0.04132231404958678 | 0.06611570247933884 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.000735 | -0.000122 |
| employment_type | salaried | 33473 | 4432 | +0.000584 | -0.000150 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000547 | -0.000097 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000371 | +0.000005 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000230 | -0.000116 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000141 | -0.000130 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000073 | -0.000140 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000011 | -0.000198 |
| employment_type | self_employed | 20987 | 2895 | -0.000108 | -0.000107 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000210 | -0.000117 |
| aux_account_months | zero | 21000 | 2461 | -0.000230 | -0.000081 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000349 | -0.000088 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000658 | -0.000154 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000939 | -0.000162 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001603 | -0.001038 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.001782 | -0.001269 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001801 | -0.000377 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002487 | -0.000482 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.