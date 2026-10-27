# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -10 | 232 | 0.3232758620689655 | 0.36637931034482757 |
| 0.200000 | 6 | 346 | 0.23121387283236994 | 0.2138728323699422 |
| 0.323686 | -19 | 405 | 0.0962962962962963 | 0.14320987654320988 |
| 0.400000 | -3 | 447 | 0.08501118568232663 | 0.09172259507829977 |
| 0.500000 | -1 | 426 | 0.07511737089201878 | 0.07746478873239436 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.006108 | +0.001296 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.004360 | +0.000128 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002914 | +0.000132 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.002630 | +0.000065 |
| employment_type | salaried | 33473 | 4432 | +0.002133 | +0.000149 |
| baseline_risk_band | under005 | 29853 | 535 | +0.002087 | +0.001431 |
| employment_type | self_employed | 20987 | 2895 | +0.001273 | +0.000024 |
| employment_type | variable_income | 15540 | 2180 | +0.000904 | +0.000367 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000472 | +0.000591 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000239 | +0.000086 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000220 | +0.000124 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000218 | +0.000153 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000188 | +0.000149 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000396 | +0.000188 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000499 | +0.000410 |
| aux_account_months | zero | 21000 | 2461 | -0.001730 | +0.000363 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001773 | +0.000056 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002096 | +0.000258 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002643 | +0.000197 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002792 | +0.000145 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003278 | +0.000007 |
| orders90_exposure_band | zero | 3428 | 106 | -0.006623 | +0.000094 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.