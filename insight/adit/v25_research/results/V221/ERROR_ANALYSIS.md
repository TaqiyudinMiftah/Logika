# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -6 | 245 | 0.33877551020408164 | 0.363265306122449 |
| 0.200000 | 2 | 358 | 0.22905027932960895 | 0.22346368715083798 |
| 0.323686 | -17 | 416 | 0.10096153846153846 | 0.14182692307692307 |
| 0.400000 | -6 | 455 | 0.08571428571428572 | 0.0989010989010989 |
| 0.500000 | -3 | 432 | 0.06944444444444445 | 0.0763888888888889 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.004132 | +0.000249 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.003068 | -0.000011 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002914 | -0.000023 |
| employment_type | salaried | 33473 | 4432 | +0.002760 | +0.000066 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002712 | +0.000850 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001671 | +0.001036 |
| employment_type | self_employed | 20987 | 2895 | +0.001646 | -0.000029 |
| employment_type | variable_income | 15540 | 2180 | +0.001280 | +0.000322 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.001130 | +0.000099 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000964 | +0.000014 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000835 | +0.000049 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000115 | +0.000306 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000524 | -0.000316 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000528 | +0.000295 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000538 | +0.000074 |
| aux_account_months | zero | 21000 | 2461 | -0.001259 | +0.000298 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001411 | -0.000005 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002543 | +0.000170 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002998 | -0.000054 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003086 | +0.000131 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003100 | +0.000079 |
| orders90_exposure_band | zero | 3428 | 106 | -0.006918 | -0.000125 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.