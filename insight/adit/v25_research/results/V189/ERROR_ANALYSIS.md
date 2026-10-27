# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 27 | 0.4444444444444444 | 0.48148148148148145 |
| 0.200000 | 4 | 42 | 0.23809523809523808 | 0.14285714285714285 |
| 0.323686 | 1 | 49 | 0.12244897959183673 | 0.10204081632653061 |
| 0.400000 | -2 | 50 | 0.08 | 0.12 |
| 0.500000 | 0 | 45 | 0.022222222222222223 | 0.022222222222222223 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.001035 | -0.000006 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000623 | +0.000008 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000544 | -0.000115 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000440 | +0.000020 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.000365 | +0.000016 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000327 | -0.000018 |
| employment_type | self_employed | 20987 | 2895 | +0.000304 | +0.000008 |
| employment_type | salaried | 33473 | 4432 | +0.000275 | -0.000014 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000208 | +0.000245 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000128 | +0.000034 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000000 | +0.000051 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000024 | -0.000019 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000089 | +0.000042 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000135 | -0.000036 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000141 | +0.000012 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000157 | +0.000000 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000163 | -0.000034 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000269 | +0.000000 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000340 | -0.000009 |
| aux_account_months | zero | 21000 | 2461 | -0.000349 | -0.000050 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000493 | +0.000030 |
| orders90_exposure_band | zero | 3428 | 106 | -0.003913 | -0.000156 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.