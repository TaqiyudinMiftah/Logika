# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 110 | 0.39090909090909093 | 0.38181818181818183 |
| 0.200000 | 0 | 151 | 0.23841059602649006 | 0.23841059602649006 |
| 0.323686 | -10 | 188 | 0.12234042553191489 | 0.17553191489361702 |
| 0.400000 | -1 | 215 | 0.11162790697674418 | 0.11627906976744186 |
| 0.500000 | 1 | 207 | 0.04830917874396135 | 0.043478260869565216 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.002308 | -0.000715 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002054 | -0.000489 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001972 | -0.000599 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000509 | -0.000129 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000502 | -0.000206 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000352 | -0.000184 |
| employment_type | salaried | 33473 | 4432 | +0.000161 | -0.000142 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000067 | -0.000186 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000000 | +0.000094 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000389 | -0.000920 |
| employment_type | variable_income | 15540 | 2180 | -0.000625 | -0.000077 |
| aux_account_months | zero | 21000 | 2461 | -0.000737 | -0.000081 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000941 | -0.000162 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000945 | -0.000177 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001046 | +0.000038 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001092 | -0.000238 |
| employment_type | self_employed | 20987 | 2895 | -0.001160 | -0.000231 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001272 | -0.000141 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001425 | -0.000183 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001922 | -0.000156 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002395 | -0.001133 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003597 | -0.000097 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.