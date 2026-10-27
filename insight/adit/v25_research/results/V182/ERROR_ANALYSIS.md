# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 4 | 0.25 | 0.5 |
| 0.200000 | 0 | 8 | 0.25 | 0.25 |
| 0.323686 | 0 | 6 | 0.0 | 0.0 |
| 0.400000 | 0 | 7 | 0.0 | 0.0 |
| 0.500000 | 1 | 6 | 0.16666666666666666 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.000472 | -0.000007 |
| employment_type | variable_income | 15540 | 2180 | +0.000212 | -0.000007 |
| employment_type | self_employed | 20987 | 2895 | +0.000193 | +0.000000 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000190 | -0.000019 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000156 | +0.000005 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000114 | +0.000006 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000070 | +0.000003 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000068 | +0.000006 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000031 | -0.000004 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000000 | +0.000011 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000000 | -0.000005 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000000 | +0.000002 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000000 | -0.000048 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000026 | +0.000003 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000028 | -0.000009 |
| employment_type | salaried | 33473 | 4432 | -0.000038 | +0.000001 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000047 | -0.000008 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000105 | -0.000008 |
| aux_account_months | zero | 21000 | 2461 | -0.000191 | -0.000009 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000223 | +0.000002 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000273 | +0.000031 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000288 | -0.000002 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.