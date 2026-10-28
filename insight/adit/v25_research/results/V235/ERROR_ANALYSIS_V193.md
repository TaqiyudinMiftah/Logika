# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 249 | 0.3815261044176707 | 0.3413654618473896 |
| 0.200000 | -4 | 366 | 0.2185792349726776 | 0.22950819672131148 |
| 0.323686 | 12 | 432 | 0.1412037037037037 | 0.11342592592592593 |
| 0.400000 | 5 | 468 | 0.08974358974358974 | 0.07905982905982906 |
| 0.500000 | 2 | 442 | 0.0746606334841629 | 0.07013574660633484 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.002439 | -0.000093 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002398 | -0.000051 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002072 | -0.000103 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000674 | +0.000043 |
| aux_account_months | zero | 21000 | 2461 | +0.000522 | -0.000241 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000015 | -0.000277 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000398 | -0.000022 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000424 | +0.000656 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000580 | -0.000004 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000670 | -0.000076 |
| employment_type | variable_income | 15540 | 2180 | -0.000679 | -0.000339 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000757 | +0.000172 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000772 | -0.000077 |
| employment_type | self_employed | 20987 | 2895 | -0.001328 | +0.000064 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001967 | -0.001120 |
| employment_type | salaried | 33473 | 4432 | -0.002405 | -0.000036 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003172 | +0.000078 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003773 | -0.000440 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.