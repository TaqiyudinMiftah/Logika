# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -7 | 251 | 0.3346613545816733 | 0.36254980079681276 |
| 0.200000 | 5 | 368 | 0.23369565217391305 | 0.22010869565217392 |
| 0.323686 | -18 | 430 | 0.10232558139534884 | 0.14418604651162792 |
| 0.400000 | -1 | 464 | 0.09051724137931035 | 0.09267241379310345 |
| 0.500000 | -1 | 451 | 0.07538802660753881 | 0.07760532150776053 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.006123 | +0.000268 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.003110 | -0.000091 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002804 | +0.000719 |
| employment_type | salaried | 33473 | 4432 | +0.002458 | +0.000018 |
| employment_type | variable_income | 15540 | 2180 | +0.001347 | +0.000399 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001247 | -0.000105 |
| employment_type | self_employed | 20987 | 2895 | +0.000610 | -0.000171 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000557 | +0.000051 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000386 | -0.000042 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000220 | -0.000009 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000192 | -0.000250 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000119 | +0.000241 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000573 | +0.000055 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000667 | -0.000155 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001327 | -0.000140 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001658 | +0.000410 |
| aux_account_months | zero | 21000 | 2461 | -0.001791 | +0.000276 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002440 | +0.000155 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003549 | +0.000040 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003575 | +0.000096 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003673 | -0.000036 |
| orders90_exposure_band | zero | 3428 | 106 | -0.007821 | -0.000222 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.