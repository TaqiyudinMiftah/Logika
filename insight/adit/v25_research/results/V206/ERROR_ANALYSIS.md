# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 12 | 256 | 0.390625 | 0.34375 |
| 0.200000 | 24 | 406 | 0.25862068965517243 | 0.19950738916256158 |
| 0.323686 | -21 | 555 | 0.12252252252252252 | 0.16036036036036036 |
| 0.400000 | 4 | 588 | 0.09523809523809523 | 0.08843537414965986 |
| 0.500000 | -12 | 595 | 0.052100840336134456 | 0.07226890756302522 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.009229 | +0.001942 |
| baseline_risk_band | under005 | 29853 | 535 | +0.008391 | +0.004124 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.008277 | +0.001520 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.004190 | -0.001608 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.003736 | +0.000741 |
| employment_type | salaried | 33473 | 4432 | +0.001895 | +0.000601 |
| aux_account_months | zero | 21000 | 2461 | +0.000787 | +0.000662 |
| employment_type | self_employed | 20987 | 2895 | +0.000707 | +0.000153 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000677 | +0.000418 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000470 | +0.000517 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000414 | +0.000372 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000325 | +0.002638 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000681 | +0.000232 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000772 | +0.000859 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000840 | +0.000382 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000906 | +0.000058 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001050 | +0.000489 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001182 | +0.000554 |
| employment_type | variable_income | 15540 | 2180 | -0.001505 | +0.000193 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001559 | +0.001666 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001758 | +0.000265 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.004153 | -0.000471 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.