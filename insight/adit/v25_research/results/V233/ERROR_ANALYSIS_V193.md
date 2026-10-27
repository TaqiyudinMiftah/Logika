# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 11 | 234 | 0.38461538461538464 | 0.33760683760683763 |
| 0.200000 | -1 | 363 | 0.22589531680440772 | 0.22865013774104684 |
| 0.323686 | 6 | 419 | 0.1288782816229117 | 0.11455847255369929 |
| 0.400000 | 5 | 451 | 0.09090909090909091 | 0.07982261640798226 |
| 0.500000 | 5 | 430 | 0.07441860465116279 | 0.06279069767441861 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002826 | +0.000034 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002758 | +0.000011 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002333 | -0.000132 |
| aux_account_months | zero | 21000 | 2461 | +0.001227 | -0.000156 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001209 | +0.000041 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000856 | +0.000813 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000112 | +0.000532 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000258 | +0.000027 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000728 | +0.000047 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000783 | +0.000027 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000791 | -0.000005 |
| employment_type | variable_income | 15540 | 2180 | -0.000979 | -0.000289 |
| employment_type | self_employed | 20987 | 2895 | -0.001093 | +0.000066 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001105 | -0.000047 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001125 | -0.001002 |
| employment_type | salaried | 33473 | 4432 | -0.002583 | +0.000034 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002602 | +0.000041 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004437 | -0.000403 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.