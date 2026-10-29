# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 246 | 0.36585365853658536 | 0.34146341463414637 |
| 0.200000 | -5 | 359 | 0.22284122562674094 | 0.23676880222841226 |
| 0.323686 | 17 | 426 | 0.1431924882629108 | 0.10328638497652583 |
| 0.400000 | 3 | 460 | 0.0891304347826087 | 0.08260869565217391 |
| 0.500000 | 2 | 445 | 0.07640449438202247 | 0.07191011235955057 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003403 | -0.000130 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003383 | -0.000081 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002528 | -0.000165 |
| aux_account_months | zero | 21000 | 2461 | +0.001845 | -0.000310 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001322 | +0.000049 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000595 | -0.000067 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000510 | +0.000248 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000307 | +0.000394 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000066 | -0.000031 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000175 | +0.000008 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000393 | -0.000080 |
| employment_type | self_employed | 20987 | 2895 | -0.000872 | +0.000048 |
| employment_type | variable_income | 15540 | 2180 | -0.001211 | -0.000333 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001680 | -0.000177 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002032 | +0.000060 |
| employment_type | salaried | 33473 | 4432 | -0.002442 | -0.000047 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002813 | -0.001373 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.006123 | -0.000193 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.