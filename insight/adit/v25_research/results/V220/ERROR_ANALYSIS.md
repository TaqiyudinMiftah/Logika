# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -8 | 243 | 0.3292181069958848 | 0.36213991769547327 |
| 0.200000 | 6 | 349 | 0.23782234957020057 | 0.22063037249283668 |
| 0.323686 | -17 | 402 | 0.09701492537313433 | 0.13930348258706468 |
| 0.400000 | -2 | 449 | 0.08908685968819599 | 0.0935412026726058 |
| 0.500000 | -3 | 423 | 0.06855791962174941 | 0.07565011820330969 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.005239 | +0.001013 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003754 | +0.000147 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.003140 | +0.000028 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.003129 | +0.000021 |
| employment_type | salaried | 33473 | 4432 | +0.002626 | +0.000091 |
| employment_type | variable_income | 15540 | 2180 | +0.001664 | +0.000334 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001255 | +0.001147 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.001188 | +0.000137 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000984 | +0.000090 |
| employment_type | self_employed | 20987 | 2895 | +0.000955 | +0.000022 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000783 | +0.000050 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000195 | +0.000499 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000100 | +0.000105 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000968 | -0.000189 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001272 | +0.000352 |
| aux_account_months | zero | 21000 | 2461 | -0.001288 | +0.000323 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001373 | +0.000048 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001839 | +0.000184 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002712 | +0.000159 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002795 | +0.000100 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003226 | -0.000040 |
| orders90_exposure_band | zero | 3428 | 106 | -0.005112 | -0.000014 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.