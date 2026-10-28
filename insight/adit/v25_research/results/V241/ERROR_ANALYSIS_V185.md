# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 251 | 0.3705179282868526 | 0.3346613545816733 |
| 0.200000 | -6 | 375 | 0.21333333333333335 | 0.22933333333333333 |
| 0.323686 | 10 | 437 | 0.13501144164759726 | 0.11212814645308924 |
| 0.400000 | -2 | 468 | 0.08547008547008547 | 0.08974358974358974 |
| 0.500000 | 5 | 453 | 0.0772626931567329 | 0.06622516556291391 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003740 | -0.000125 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003691 | -0.000157 |
| aux_account_months | zero | 21000 | 2461 | +0.001953 | -0.000332 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000755 | -0.000173 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000441 | -0.000103 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000151 | -0.000005 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000292 | -0.000254 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000661 | -0.000106 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000752 | -0.000065 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000806 | +0.000309 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000871 | -0.000026 |
| employment_type | self_employed | 20987 | 2895 | -0.001078 | +0.000047 |
| employment_type | variable_income | 15540 | 2180 | -0.001326 | -0.000411 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.002072 | +0.000069 |
| employment_type | salaried | 33473 | 4432 | -0.002320 | -0.000074 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002446 | +0.000053 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002573 | -0.001675 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.008475 | -0.000128 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.