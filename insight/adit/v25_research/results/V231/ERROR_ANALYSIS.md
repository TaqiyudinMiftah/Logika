# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 3 | 23 | 0.4782608695652174 | 0.34782608695652173 |
| 0.200000 | -2 | 45 | 0.2 | 0.24444444444444444 |
| 0.323686 | -3 | 40 | 0.125 | 0.2 |
| 0.400000 | 1 | 52 | 0.17307692307692307 | 0.15384615384615385 |
| 0.500000 | 0 | 51 | 0.0392156862745098 | 0.0392156862745098 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.002107 | +0.000023 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001744 | -0.000128 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001525 | +0.000109 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000901 | +0.000279 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000690 | +0.000009 |
| employment_type | self_employed | 20987 | 2895 | +0.000635 | +0.000006 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000450 | +0.000033 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000274 | -0.000004 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000230 | +0.000089 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000220 | +0.000067 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000086 | +0.000001 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.000085 | +0.000064 |
| aux_account_months | zero | 21000 | 2461 | +0.000079 | +0.000099 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000021 | -0.000003 |
| employment_type | variable_income | 15540 | 2180 | -0.000113 | +0.000034 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000124 | +0.000032 |
| employment_type | salaried | 33473 | 4432 | -0.000309 | +0.000030 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000463 | -0.000006 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000650 | +0.000084 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000758 | +0.000024 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000805 | +0.000246 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000982 | +0.000022 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.