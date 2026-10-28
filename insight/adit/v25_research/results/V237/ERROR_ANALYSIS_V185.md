# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 253 | 0.391304347826087 | 0.3557312252964427 |
| 0.200000 | 4 | 375 | 0.23733333333333334 | 0.22666666666666666 |
| 0.323686 | 16 | 447 | 0.14988814317673377 | 0.11409395973154363 |
| 0.400000 | -2 | 472 | 0.07838983050847458 | 0.0826271186440678 |
| 0.500000 | -1 | 471 | 0.06369426751592357 | 0.06581740976645435 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002162 | -0.000024 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001884 | -0.000025 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.001586 | +0.000683 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001149 | -0.000145 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000702 | -0.000055 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000677 | -0.000135 |
| aux_account_months | zero | 21000 | 2461 | +0.000313 | -0.000177 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000191 | -0.000164 |
| employment_type | variable_income | 15540 | 2180 | +0.000082 | -0.000313 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000034 | -0.000124 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000203 | -0.000109 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000334 | -0.000124 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000346 | -0.000139 |
| employment_type | salaried | 33473 | 4432 | -0.001210 | -0.000123 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001231 | +0.000079 |
| employment_type | self_employed | 20987 | 2895 | -0.001756 | -0.000019 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002224 | -0.002131 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003924 | -0.000885 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.