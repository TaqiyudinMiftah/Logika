# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 371 | 0.37735849056603776 | 0.3665768194070081 |
| 0.200000 | 4 | 578 | 0.2370242214532872 | 0.2301038062283737 |
| 0.323686 | -1 | 721 | 0.12482662968099861 | 0.1262135922330097 |
| 0.400000 | 0 | 805 | 0.09440993788819876 | 0.09440993788819876 |
| 0.500000 | -18 | 831 | 0.06016847172081829 | 0.08182912154031288 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.003448 | -0.000872 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000351 | -0.000584 |
| employment_type | salaried | 33473 | 4432 | -0.000728 | -0.000964 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001082 | -0.001003 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001318 | -0.002691 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001542 | -0.000423 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001683 | -0.000952 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001719 | -0.000933 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001926 | -0.001009 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002413 | -0.000653 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.002427 | -0.001138 |
| employment_type | self_employed | 20987 | 2895 | -0.002529 | -0.000801 |
| employment_type | variable_income | 15540 | 2180 | -0.002866 | -0.000845 |
| aux_account_months | zero | 21000 | 2461 | -0.003721 | -0.000545 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004215 | -0.001327 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.007183 | -0.008364 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.010806 | -0.002823 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.014654 | -0.004100 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.