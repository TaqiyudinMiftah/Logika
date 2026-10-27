# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 31 | 0.41935483870967744 | 0.3548387096774194 |
| 0.200000 | -3 | 58 | 0.1896551724137931 | 0.2413793103448276 |
| 0.323686 | -5 | 55 | 0.10909090909090909 | 0.2 |
| 0.400000 | 1 | 75 | 0.14666666666666667 | 0.13333333333333333 |
| 0.500000 | 0 | 72 | 0.027777777777777776 | 0.027777777777777776 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.003010 | -0.000068 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.002061 | +0.000352 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001744 | -0.000145 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001179 | +0.000136 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000616 | -0.000002 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000471 | +0.000040 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000258 | +0.000008 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000165 | +0.000086 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000145 | +0.000116 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000059 | -0.000013 |
| employment_type | salaried | 33473 | 4432 | +0.000043 | +0.000041 |
| employment_type | self_employed | 20987 | 2895 | -0.000083 | +0.000005 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000164 | +0.000309 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000237 | +0.000040 |
| aux_account_months | zero | 21000 | 2461 | -0.000252 | +0.000131 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000332 | -0.000006 |
| employment_type | variable_income | 15540 | 2180 | -0.000332 | +0.000039 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000398 | -0.000001 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000665 | +0.000085 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001003 | +0.000025 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001107 | +0.000025 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001348 | +0.000096 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.