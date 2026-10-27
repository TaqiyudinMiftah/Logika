# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 100 | 0.35 | 0.34 |
| 0.200000 | 8 | 164 | 0.25 | 0.20121951219512196 |
| 0.323686 | -13 | 197 | 0.09137055837563451 | 0.15736040609137056 |
| 0.400000 | -4 | 222 | 0.08108108108108109 | 0.0990990990990991 |
| 0.500000 | 2 | 235 | 0.06808510638297872 | 0.059574468085106386 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.004853 | -0.000516 |
| employment_type | self_employed | 20987 | 2895 | +0.001094 | +0.000082 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000839 | +0.001061 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000807 | -0.002019 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000508 | +0.000164 |
| employment_type | variable_income | 15540 | 2180 | +0.000252 | +0.000230 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000188 | +0.000074 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000086 | +0.000124 |
| aux_account_months | zero | 21000 | 2461 | +0.000022 | +0.000097 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000037 | +0.000078 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000252 | +0.000081 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000259 | +0.000121 |
| employment_type | salaried | 33473 | 4432 | -0.000336 | -0.000036 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000833 | -0.000063 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000996 | +0.000230 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001115 | +0.000100 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001246 | +0.000063 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001692 | +0.000094 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002164 | -0.000258 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002408 | -0.000098 |
| baseline_risk_band | under005 | 29853 | 535 | -0.002592 | -0.000977 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003046 | +0.000039 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.