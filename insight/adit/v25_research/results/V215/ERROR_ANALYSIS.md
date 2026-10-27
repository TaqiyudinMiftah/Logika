# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 11 | 154 | 0.38961038961038963 | 0.3181818181818182 |
| 0.200000 | 9 | 236 | 0.2288135593220339 | 0.1906779661016949 |
| 0.323686 | -27 | 309 | 0.0970873786407767 | 0.18446601941747573 |
| 0.400000 | -18 | 322 | 0.07142857142857142 | 0.12732919254658384 |
| 0.500000 | -11 | 346 | 0.046242774566473986 | 0.07803468208092486 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.007547 | +0.000084 |
| baseline_risk_band | under005 | 29853 | 535 | +0.005635 | +0.001935 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002047 | +0.000416 |
| employment_type | self_employed | 20987 | 2895 | +0.001799 | +0.000184 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.001080 | +0.000603 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001061 | -0.000465 |
| employment_type | salaried | 33473 | 4432 | +0.000756 | +0.000121 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000596 | +0.000525 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000210 | +0.000322 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000242 | +0.000320 |
| aux_account_months | zero | 21000 | 2461 | -0.000374 | +0.000315 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000611 | +0.000190 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000713 | +0.000070 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000962 | +0.000056 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001007 | +0.001293 |
| employment_type | variable_income | 15540 | 2180 | -0.001347 | +0.000230 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001389 | +0.000136 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001528 | -0.000081 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001970 | +0.000113 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003174 | -0.000224 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004640 | +0.000267 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.005187 | +0.000829 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.