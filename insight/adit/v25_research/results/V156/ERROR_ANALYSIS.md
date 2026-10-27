# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 12 | 0.5 | 0.4166666666666667 |
| 0.200000 | 1 | 20 | 0.15 | 0.1 |
| 0.323686 | -4 | 21 | 0.047619047619047616 | 0.23809523809523808 |
| 0.400000 | 4 | 15 | 0.3333333333333333 | 0.06666666666666667 |
| 0.500000 | 2 | 23 | 0.08695652173913043 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.000329 | -0.000157 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000239 | -0.000107 |
| employment_type | variable_income | 15540 | 2180 | +0.000215 | -0.000029 |
| employment_type | self_employed | 20987 | 2895 | +0.000193 | -0.000027 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000081 | -0.000034 |
| employment_type | salaried | 33473 | 4432 | +0.000033 | -0.000039 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000020 | -0.000026 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000049 | -0.000037 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000131 | -0.000030 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000245 | -0.000033 |
| aux_account_months | zero | 21000 | 2461 | -0.000248 | -0.000018 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000293 | -0.000036 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000355 | -0.000035 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000586 | -0.000012 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000612 | -0.000054 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001170 | -0.000193 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001612 | +0.000000 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.002150 | -0.000245 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.