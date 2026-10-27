# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 18 | 282 | 0.39361702127659576 | 0.32978723404255317 |
| 0.200000 | 5 | 455 | 0.22637362637362637 | 0.2153846153846154 |
| 0.323686 | -10 | 546 | 0.11538461538461539 | 0.1336996336996337 |
| 0.400000 | -14 | 566 | 0.07420494699646643 | 0.0989399293286219 |
| 0.500000 | -9 | 575 | 0.06260869565217392 | 0.0782608695652174 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29965 | 546 | +0.004522 | +0.001267 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.004352 | -0.000817 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.004350 | +0.000258 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003593 | +0.000241 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003161 | +0.000188 |
| aux_account_months | zero | 21000 | 2461 | +0.001471 | +0.000005 |
| employment_type | self_employed | 20987 | 2895 | +0.000913 | +0.000231 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000108 | +0.000126 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000205 | -0.000043 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000469 | +0.001500 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001514 | -0.000021 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001586 | +0.000108 |
| employment_type | salaried | 33473 | 4432 | -0.001804 | +0.000076 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002264 | +0.000122 |
| employment_type | variable_income | 15540 | 2180 | -0.002470 | -0.000104 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005062 | -0.000692 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.005426 | +0.001457 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.006828 | +0.000313 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.