# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 274 | 0.3795620437956204 | 0.354014598540146 |
| 0.200000 | -2 | 412 | 0.22815533980582525 | 0.23300970873786409 |
| 0.323686 | 9 | 484 | 0.1322314049586777 | 0.11363636363636363 |
| 0.400000 | 3 | 525 | 0.09523809523809523 | 0.08952380952380952 |
| 0.500000 | 7 | 525 | 0.07809523809523809 | 0.06476190476190476 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003461 | -0.000331 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003417 | -0.000312 |
| aux_account_months | zero | 21000 | 2461 | +0.000918 | -0.000432 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000062 | -0.000763 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000037 | -0.000131 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000144 | -0.000008 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000568 | -0.000281 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.000884 | -0.001339 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001192 | -0.000163 |
| employment_type | salaried | 33473 | 4432 | -0.001725 | -0.000179 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002077 | -0.000132 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002129 | -0.000099 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002142 | -0.000137 |
| employment_type | self_employed | 20987 | 2895 | -0.002267 | -0.000187 |
| employment_type | variable_income | 15540 | 2180 | -0.002333 | -0.000380 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002376 | -0.000168 |
| baseline_risk_band | under005 | 29965 | 546 | -0.002771 | -0.001123 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003905 | -0.000778 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.