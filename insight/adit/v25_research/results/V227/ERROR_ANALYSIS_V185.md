# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 245 | 0.3795918367346939 | 0.34285714285714286 |
| 0.200000 | -7 | 364 | 0.21703296703296704 | 0.23626373626373626 |
| 0.323686 | 12 | 418 | 0.14114832535885166 | 0.11244019138755981 |
| 0.400000 | 5 | 457 | 0.08533916849015317 | 0.07439824945295405 |
| 0.500000 | 2 | 445 | 0.07415730337078652 | 0.0696629213483146 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003548 | +0.000004 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003548 | -0.000017 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.002182 | +0.000800 |
| aux_account_months | zero | 21000 | 2461 | +0.001594 | -0.000179 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001197 | -0.000132 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000320 | +0.000062 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000266 | -0.000024 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000293 | -0.000069 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000595 | -0.000029 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000626 | +0.000003 |
| employment_type | self_employed | 20987 | 2895 | -0.000969 | +0.000052 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001081 | +0.000359 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001237 | +0.000081 |
| employment_type | variable_income | 15540 | 2180 | -0.001454 | -0.000295 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002129 | +0.000033 |
| employment_type | salaried | 33473 | 4432 | -0.002518 | -0.000004 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.003044 | -0.001225 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004379 | -0.000373 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.