# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 4 | 0.5 | 0.75 |
| 0.200000 | -1 | 4 | 0.25 | 0.5 |
| 0.323686 | 0 | 5 | 0.0 | 0.0 |
| 0.400000 | -1 | 4 | 0.0 | 0.25 |
| 0.500000 | 0 | 4 | 0.0 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.000301 | +0.000028 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000224 | -0.000008 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000201 | -0.000069 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000188 | -0.000007 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000158 | -0.000003 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000156 | +0.000013 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000131 | -0.000003 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000119 | -0.000001 |
| employment_type | salaried | 33473 | 4432 | +0.000119 | -0.000002 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000091 | -0.000003 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000057 | -0.000003 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000034 | +0.000043 |
| employment_type | self_employed | 20987 | 2895 | +0.000014 | +0.000001 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000001 | -0.000003 |
| aux_account_months | zero | 21000 | 2461 | +0.000000 | +0.000000 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000000 | +0.000034 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000000 | +0.000002 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000000 | +0.000011 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000000 | +0.000001 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000058 | -0.000009 |
| employment_type | variable_income | 15540 | 2180 | -0.000088 | +0.000001 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000407 | -0.000003 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.