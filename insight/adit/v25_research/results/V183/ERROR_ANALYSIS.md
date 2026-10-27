# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 33 | 0.3333333333333333 | 0.3333333333333333 |
| 0.200000 | 1 | 45 | 0.2222222222222222 | 0.2 |
| 0.323686 | -5 | 39 | 0.10256410256410256 | 0.23076923076923078 |
| 0.400000 | 1 | 47 | 0.10638297872340426 | 0.0851063829787234 |
| 0.500000 | 2 | 52 | 0.038461538461538464 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.001675 | +0.000003 |
| employment_type | variable_income | 15540 | 2180 | +0.000608 | -0.000003 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000326 | +0.000010 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000297 | -0.000055 |
| aux_account_months | zero | 21000 | 2461 | +0.000248 | -0.000049 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000194 | -0.000040 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000039 | +0.000002 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000009 | -0.000009 |
| employment_type | salaried | 33473 | 4432 | -0.000019 | -0.000027 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000100 | +0.000007 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000198 | +0.000013 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000249 | -0.000003 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000258 | +0.000015 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000277 | +0.000008 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000291 | -0.000021 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000450 | -0.000014 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000474 | -0.000050 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000480 | -0.000058 |
| employment_type | self_employed | 20987 | 2895 | -0.000636 | +0.000013 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001237 | -0.000139 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001305 | +0.000008 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002107 | +0.000094 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.