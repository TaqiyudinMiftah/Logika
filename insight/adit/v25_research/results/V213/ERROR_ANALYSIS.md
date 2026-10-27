# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -6 | 225 | 0.3377777777777778 | 0.36444444444444446 |
| 0.200000 | 5 | 330 | 0.23030303030303031 | 0.21515151515151515 |
| 0.323686 | -14 | 389 | 0.10025706940874037 | 0.13624678663239073 |
| 0.400000 | -1 | 426 | 0.0892018779342723 | 0.09154929577464789 |
| 0.500000 | -2 | 405 | 0.07160493827160494 | 0.07654320987654321 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.007557 | +0.001306 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003754 | +0.000184 |
| baseline_risk_band | under005 | 29853 | 535 | +0.002811 | +0.001388 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002705 | +0.000136 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.002261 | +0.000070 |
| employment_type | salaried | 33473 | 4432 | +0.002022 | +0.000149 |
| employment_type | self_employed | 20987 | 2895 | +0.001328 | +0.000047 |
| employment_type | variable_income | 15540 | 2180 | +0.001017 | +0.000372 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000572 | +0.000653 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000319 | +0.000128 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000318 | +0.000095 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000002 | +0.000155 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000024 | +0.000150 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000338 | +0.000419 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000941 | +0.000272 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001638 | +0.000072 |
| aux_account_months | zero | 21000 | 2461 | -0.001673 | +0.000361 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002096 | +0.000266 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003050 | +0.000207 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003249 | +0.000163 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003842 | +0.000002 |
| orders90_exposure_band | zero | 3428 | 106 | -0.004214 | -0.000006 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.