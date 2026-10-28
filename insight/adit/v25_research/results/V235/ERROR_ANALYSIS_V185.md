# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 250 | 0.38 | 0.344 |
| 0.200000 | -4 | 359 | 0.21448467966573817 | 0.22562674094707522 |
| 0.323686 | 17 | 433 | 0.14780600461893764 | 0.10854503464203233 |
| 0.400000 | 2 | 462 | 0.08441558441558442 | 0.08008658008658008 |
| 0.500000 | 2 | 449 | 0.0757238307349666 | 0.07126948775055679 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.002950 | -0.000116 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002629 | -0.000068 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001848 | -0.000101 |
| aux_account_months | zero | 21000 | 2461 | +0.000964 | -0.000273 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000450 | +0.000058 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000094 | -0.000021 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000057 | +0.000004 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000230 | -0.000077 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000339 | -0.000084 |
| employment_type | variable_income | 15540 | 2180 | -0.000370 | -0.000333 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000378 | +0.000195 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000542 | +0.000653 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000770 | -0.000260 |
| employment_type | self_employed | 20987 | 2895 | -0.000969 | +0.000066 |
| employment_type | salaried | 33473 | 4432 | -0.002278 | -0.000040 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003119 | +0.000096 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.003345 | -0.001186 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005934 | -0.000340 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.