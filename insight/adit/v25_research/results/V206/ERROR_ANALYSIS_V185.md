# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 19 | 281 | 0.40569395017793597 | 0.33807829181494664 |
| 0.200000 | 20 | 474 | 0.2489451476793249 | 0.20675105485232068 |
| 0.323686 | -4 | 592 | 0.1266891891891892 | 0.13344594594594594 |
| 0.400000 | 8 | 600 | 0.1 | 0.08666666666666667 |
| 0.500000 | -10 | 610 | 0.060655737704918035 | 0.07704918032786885 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29965 | 546 | +0.007393 | +0.003272 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.004691 | +0.000874 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.004060 | +0.000336 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003873 | +0.000385 |
| aux_account_months | zero | 21000 | 2461 | +0.002633 | +0.000353 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001122 | +0.000397 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000417 | +0.000095 |
| employment_type | self_employed | 20987 | 2895 | -0.000179 | +0.000201 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000546 | +0.000426 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000571 | +0.002590 |
| employment_type | salaried | 33473 | 4432 | -0.000666 | +0.000556 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001037 | +0.000354 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001232 | +0.000155 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001240 | +0.001765 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001933 | -0.001836 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002052 | +0.000274 |
| employment_type | variable_income | 15540 | 2180 | -0.002627 | -0.000141 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002960 | +0.000905 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.