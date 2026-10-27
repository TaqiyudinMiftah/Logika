# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -3 | 11 | 0.2727272727272727 | 0.5454545454545454 |
| 0.200000 | 2 | 17 | 0.29411764705882354 | 0.17647058823529413 |
| 0.323686 | -2 | 25 | 0.08 | 0.16 |
| 0.400000 | 1 | 30 | 0.1 | 0.06666666666666667 |
| 0.500000 | -1 | 30 | 0.0 | 0.03333333333333333 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000743 | +0.000264 |
| employment_type | variable_income | 15540 | 2180 | +0.000485 | +0.000048 |
| employment_type | self_employed | 20987 | 2895 | +0.000484 | +0.000046 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000133 | +0.000040 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000107 | +0.000211 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000019 | +0.000037 |
| employment_type | salaried | 33473 | 4432 | -0.000045 | +0.000011 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000072 | +0.000036 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000213 | +0.000011 |
| aux_account_months | zero | 21000 | 2461 | -0.000263 | +0.000006 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000379 | +0.000024 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000385 | +0.000009 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000389 | +0.000007 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000560 | +0.000058 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000575 | +0.000035 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000628 | -0.000047 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000902 | +0.000126 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001138 | +0.000075 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.