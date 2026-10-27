# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 247 | 0.3805668016194332 | 0.3481781376518219 |
| 0.200000 | -7 | 366 | 0.22404371584699453 | 0.24316939890710382 |
| 0.323686 | 11 | 432 | 0.13657407407407407 | 0.1111111111111111 |
| 0.400000 | 7 | 462 | 0.09090909090909091 | 0.07575757575757576 |
| 0.500000 | 2 | 441 | 0.07256235827664399 | 0.06802721088435375 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002569 | -0.000047 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002259 | -0.000075 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001907 | -0.000161 |
| baseline_risk_band | under005 | 29938 | 544 | +0.001819 | -0.000256 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000846 | -0.000017 |
| aux_account_months | zero | 21000 | 2461 | +0.000619 | -0.000239 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000309 | -0.000086 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000583 | +0.000472 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000683 | -0.000118 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000718 | -0.000070 |
| employment_type | variable_income | 15540 | 2180 | -0.000819 | -0.000336 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000846 | -0.000050 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000846 | +0.000146 |
| employment_type | self_employed | 20987 | 2895 | -0.001535 | +0.000023 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002081 | -0.000023 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.002285 | -0.001655 |
| employment_type | salaried | 33473 | 4432 | -0.002558 | -0.000086 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005062 | -0.000646 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.