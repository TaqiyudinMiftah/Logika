# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 5 | 0.4 | 0.6 |
| 0.200000 | 0 | 5 | 0.0 | 0.0 |
| 0.323686 | -4 | 11 | 0.0 | 0.36363636363636365 |
| 0.400000 | 1 | 4 | 0.25 | 0.0 |
| 0.500000 | 2 | 9 | 0.2222222222222222 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.000307 | -0.000047 |
| employment_type | variable_income | 15540 | 2180 | +0.000286 | -0.000004 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000042 | -0.000038 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000003 | -0.000013 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000000 | -0.000014 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000003 | -0.000014 |
| employment_type | self_employed | 20987 | 2895 | -0.000014 | -0.000010 |
| aux_account_months | zero | 21000 | 2461 | -0.000025 | -0.000009 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000029 | -0.000098 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000039 | -0.000011 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000062 | -0.000012 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000067 | -0.000002 |
| employment_type | salaried | 33473 | 4432 | -0.000157 | -0.000016 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000211 | -0.000013 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000269 | -0.000013 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000379 | -0.000020 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000912 | -0.000069 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001138 | +0.000025 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.