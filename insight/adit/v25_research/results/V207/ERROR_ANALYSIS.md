# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -3 | 283 | 0.3710247349823322 | 0.38162544169611307 |
| 0.200000 | 8 | 451 | 0.2328159645232816 | 0.21507760532150777 |
| 0.323686 | -18 | 560 | 0.09821428571428571 | 0.13035714285714287 |
| 0.400000 | -4 | 656 | 0.0975609756097561 | 0.10365853658536585 |
| 0.500000 | -20 | 702 | 0.0584045584045584 | 0.0868945868945869 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.002126 | -0.000910 |
| employment_type | salaried | 33473 | 4432 | +0.001833 | -0.000919 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000246 | -0.002491 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001132 | -0.000875 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001523 | -0.000905 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001585 | -0.000940 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001632 | -0.001018 |
| employment_type | self_employed | 20987 | 2895 | -0.001643 | -0.000848 |
| employment_type | variable_income | 15540 | 2180 | -0.001743 | -0.000511 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002027 | -0.001373 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002519 | -0.001548 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.002982 | -0.006796 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003734 | -0.000503 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.004716 | -0.000495 |
| aux_allocation_count | zero | 23732 | 2895 | -0.004945 | -0.000291 |
| aux_account_months | zero | 21000 | 2461 | -0.005567 | -0.000235 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.008531 | -0.003872 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.010694 | -0.003410 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.