# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 160 | 0.3625 | 0.375 |
| 0.200000 | 10 | 268 | 0.23507462686567165 | 0.19776119402985073 |
| 0.323686 | -12 | 331 | 0.1148036253776435 | 0.1510574018126888 |
| 0.400000 | -7 | 367 | 0.08446866485013624 | 0.10354223433242507 |
| 0.500000 | -4 | 388 | 0.05670103092783505 | 0.06701030927835051 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.007118 | +0.002277 |
| baseline_risk_band | under005 | 29853 | 535 | +0.006849 | +0.002652 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.001792 | +0.001472 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001188 | +0.000496 |
| employment_type | variable_income | 15540 | 2180 | +0.001171 | +0.000370 |
| employment_type | salaried | 33473 | 4432 | +0.000917 | +0.000386 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.000154 | +0.000264 |
| aux_account_months | zero | 21000 | 2461 | +0.000093 | +0.000265 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000029 | +0.000471 |
| employment_type | self_employed | 20987 | 2895 | -0.000093 | +0.000198 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000188 | +0.000444 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000317 | +0.000407 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000404 | +0.000340 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000767 | +0.000222 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000904 | +0.000380 |
| orders90_exposure_band | zero | 3428 | 106 | -0.001005 | -0.000102 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001542 | +0.000322 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001801 | +0.000270 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001837 | +0.000336 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001945 | +0.000112 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002017 | +0.001006 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002232 | -0.000094 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.