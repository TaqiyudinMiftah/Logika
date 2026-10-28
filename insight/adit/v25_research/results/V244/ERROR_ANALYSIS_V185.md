# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 261 | 0.3716475095785441 | 0.3371647509578544 |
| 0.200000 | -2 | 399 | 0.22807017543859648 | 0.23308270676691728 |
| 0.323686 | 6 | 481 | 0.12474012474012475 | 0.11226611226611227 |
| 0.400000 | 2 | 503 | 0.09145129224652088 | 0.0874751491053678 |
| 0.500000 | -1 | 496 | 0.07056451612903226 | 0.07258064516129033 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6079 | 3488 | +0.004686 | +0.000140 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003154 | -0.000175 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003068 | -0.000172 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001591 | -0.000140 |
| aux_account_months | zero | 21000 | 2461 | +0.000788 | -0.000329 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000681 | -0.000169 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000859 | -0.000298 |
| employment_type | variable_income | 15540 | 2180 | -0.001483 | -0.000395 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001672 | -0.000379 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.001772 | -0.001822 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001976 | -0.000229 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002067 | -0.000159 |
| employment_type | self_employed | 20987 | 2895 | -0.002212 | -0.000144 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002242 | -0.000186 |
| baseline_risk_band | under005 | 29965 | 546 | -0.002306 | -0.000964 |
| employment_type | salaried | 33473 | 4432 | -0.002545 | -0.000168 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002807 | -0.000045 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003033 | -0.000522 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.