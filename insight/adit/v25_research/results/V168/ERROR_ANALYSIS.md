# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 2 | 0.5 | 1.0 |
| 0.200000 | 0 | 1 | 1.0 | 1.0 |
| 0.323686 | -1 | 3 | 0.0 | 0.3333333333333333 |
| 0.400000 | 0 | 7 | 0.0 | 0.0 |
| 0.500000 | 0 | 4 | 0.0 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000256 | -0.000006 |
| aux_account_months | zero | 21000 | 2461 | +0.000245 | +0.000004 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000201 | +0.000002 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000201 | +0.000009 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000181 | +0.000001 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000126 | -0.000001 |
| employment_type | self_employed | 20987 | 2895 | +0.000028 | -0.000001 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000000 | -0.000011 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000000 | +0.000008 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000019 | -0.000001 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000048 | +0.000000 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000068 | +0.000005 |
| employment_type | salaried | 33473 | 4432 | -0.000072 | +0.000000 |
| employment_type | variable_income | 15540 | 2180 | -0.000075 | +0.000003 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000079 | +0.000001 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000179 | +0.000007 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000270 | -0.000003 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000648 | +0.000001 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.