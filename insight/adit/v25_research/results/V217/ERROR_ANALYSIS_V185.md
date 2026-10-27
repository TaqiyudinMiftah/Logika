# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -3 | 31 | 0.41935483870967744 | 0.5161290322580645 |
| 0.200000 | 2 | 45 | 0.24444444444444444 | 0.2 |
| 0.323686 | -2 | 64 | 0.109375 | 0.140625 |
| 0.400000 | 1 | 51 | 0.0392156862745098 | 0.0196078431372549 |
| 0.500000 | 1 | 66 | 0.07575757575757576 | 0.06060606060606061 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.003299 | +0.000453 |
| baseline_risk_band | under005 | 29965 | 546 | +0.001425 | +0.000576 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000942 | +0.000504 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000761 | +0.000064 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000726 | +0.000178 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000692 | +0.000086 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000591 | +0.000064 |
| employment_type | self_employed | 20987 | 2895 | +0.000387 | +0.000072 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000208 | +0.000100 |
| aux_account_months | zero | 21000 | 2461 | +0.000115 | +0.000053 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000023 | +0.000096 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000055 | +0.000095 |
| employment_type | variable_income | 15540 | 2180 | -0.000219 | +0.000033 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000333 | +0.000076 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000375 | +0.000540 |
| employment_type | salaried | 33473 | 4432 | -0.000428 | +0.000105 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000451 | +0.000094 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001763 | -0.000099 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.