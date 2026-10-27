# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -9 | 273 | 0.3663003663003663 | 0.3992673992673993 |
| 0.200000 | -1 | 413 | 0.20581113801452786 | 0.20823244552058112 |
| 0.323686 | -29 | 539 | 0.10575139146567718 | 0.15955473098330242 |
| 0.400000 | 4 | 555 | 0.0972972972972973 | 0.09009009009009009 |
| 0.500000 | -20 | 579 | 0.05008635578583765 | 0.0846286701208981 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002253 | -0.003690 |
| employment_type | variable_income | 15540 | 2180 | +0.001756 | -0.000907 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000426 | -0.001103 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000241 | -0.000463 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000402 | -0.000542 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000527 | -0.000721 |
| aux_account_months | zero | 21000 | 2461 | -0.000629 | -0.000469 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000693 | -0.000723 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000730 | -0.000440 |
| employment_type | self_employed | 20987 | 2895 | -0.000979 | -0.000223 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001004 | -0.000503 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001458 | -0.000995 |
| employment_type | salaried | 33473 | 4432 | -0.001572 | -0.000424 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001739 | -0.000615 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.004680 | -0.000234 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005877 | -0.001446 |
| baseline_risk_band | under005 | 29853 | 535 | -0.005935 | +0.000217 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.008364 | -0.003047 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.