# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 250 | 0.368 | 0.352 |
| 0.200000 | -5 | 372 | 0.21505376344086022 | 0.22849462365591397 |
| 0.323686 | 10 | 426 | 0.14788732394366197 | 0.12441314553990611 |
| 0.400000 | 3 | 467 | 0.07922912205567452 | 0.0728051391862955 |
| 0.500000 | 0 | 455 | 0.07252747252747253 | 0.07252747252747253 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.002970 | -0.000084 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002923 | -0.000055 |
| aux_account_months | zero | 21000 | 2461 | +0.001180 | -0.000236 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000903 | -0.000150 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000180 | +0.000048 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000126 | -0.000065 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000264 | -0.000091 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000462 | -0.000058 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000480 | +0.000071 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000508 | -0.000103 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000598 | -0.000033 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000619 | +0.000612 |
| employment_type | variable_income | 15540 | 2180 | -0.001098 | -0.000330 |
| employment_type | self_employed | 20987 | 2895 | -0.001204 | +0.000028 |
| employment_type | salaried | 33473 | 4432 | -0.002511 | -0.000062 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002924 | +0.000022 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.003249 | -0.001470 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003716 | -0.000451 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.