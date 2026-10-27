# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 247 | 0.3724696356275304 | 0.3481781376518219 |
| 0.200000 | -2 | 378 | 0.22486772486772486 | 0.23015873015873015 |
| 0.323686 | 10 | 441 | 0.1473922902494331 | 0.12471655328798185 |
| 0.400000 | 6 | 469 | 0.08102345415778252 | 0.06823027718550106 |
| 0.500000 | 0 | 464 | 0.0668103448275862 | 0.0668103448275862 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003593 | -0.000061 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003279 | -0.000075 |
| aux_account_months | zero | 21000 | 2461 | +0.002108 | -0.000246 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000428 | -0.000085 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000037 | +0.000021 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000015 | -0.000260 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000032 | -0.000006 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000244 | +0.000145 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000422 | -0.000076 |
| employment_type | variable_income | 15540 | 2180 | -0.000434 | -0.000348 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000455 | -0.000113 |
| employment_type | self_employed | 20987 | 2895 | -0.000513 | -0.000039 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000784 | -0.000133 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001350 | -0.000090 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.001832 | +0.000284 |
| employment_type | salaried | 33473 | 4432 | -0.002500 | -0.000100 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002887 | -0.001957 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003716 | -0.000245 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.