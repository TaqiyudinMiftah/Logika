# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 228 | 0.33771929824561403 | 0.34210526315789475 |
| 0.200000 | -4 | 339 | 0.20648967551622419 | 0.2182890855457227 |
| 0.323686 | -15 | 389 | 0.10282776349614396 | 0.14138817480719795 |
| 0.400000 | 4 | 430 | 0.09767441860465116 | 0.08837209302325581 |
| 0.500000 | -4 | 411 | 0.0681265206812652 | 0.07785888077858881 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.005953 | +0.000350 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.005112 | -0.000002 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002602 | +0.001031 |
| employment_type | salaried | 33473 | 4432 | +0.002090 | +0.000071 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.001323 | +0.000337 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.001253 | -0.000015 |
| employment_type | variable_income | 15540 | 2180 | +0.001233 | +0.000381 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000866 | +0.001241 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000255 | -0.000981 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000180 | +0.000032 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000160 | -0.000137 |
| aux_account_months | zero | 21000 | 2461 | -0.000187 | +0.000337 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000551 | +0.000070 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000561 | +0.000007 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000647 | +0.000017 |
| employment_type | self_employed | 20987 | 2895 | -0.000744 | -0.000066 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000974 | +0.000170 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001242 | +0.000222 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001594 | +0.000142 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001749 | +0.000163 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003435 | -0.000145 |
| orders90_exposure_band | zero | 3428 | 106 | -0.007821 | -0.000102 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.