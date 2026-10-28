# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 58 | 0.3620689655172414 | 0.3275862068965517 |
| 0.200000 | 3 | 92 | 0.25 | 0.21739130434782608 |
| 0.323686 | -8 | 103 | 0.10679611650485436 | 0.18446601941747573 |
| 0.400000 | 2 | 123 | 0.0975609756097561 | 0.08130081300813008 |
| 0.500000 | -1 | 118 | 0.0423728813559322 | 0.05084745762711865 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.003589 | +0.000076 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002199 | -0.000251 |
| employment_type | variable_income | 15540 | 2180 | +0.001325 | -0.000005 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000903 | +0.000099 |
| employment_type | salaried | 33473 | 4432 | +0.000672 | -0.000057 |
| aux_account_months | zero | 21000 | 2461 | +0.000302 | -0.000046 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000037 | -0.000041 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000121 | -0.000089 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000295 | +0.000013 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000333 | -0.000032 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000402 | -0.000079 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000413 | -0.000480 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000470 | -0.000383 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000595 | -0.000093 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000642 | -0.000056 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000826 | -0.000061 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000946 | -0.000133 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001048 | -0.000038 |
| employment_type | self_employed | 20987 | 2895 | -0.001090 | -0.000087 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001301 | -0.000064 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001480 | -0.000702 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002080 | +0.000021 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.