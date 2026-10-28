# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 3 | 169 | 0.3609467455621302 | 0.3431952662721893 |
| 0.200000 | 19 | 277 | 0.2815884476534296 | 0.21299638989169675 |
| 0.323686 | -19 | 353 | 0.1048158640226629 | 0.15864022662889518 |
| 0.400000 | -8 | 358 | 0.08100558659217877 | 0.10335195530726257 |
| 0.500000 | -6 | 382 | 0.05235602094240838 | 0.06806282722513089 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.007311 | -0.000108 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.004234 | +0.001177 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003280 | -0.001038 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.002652 | +0.000385 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001951 | +0.001647 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001400 | +0.002251 |
| employment_type | variable_income | 15540 | 2180 | +0.001121 | -0.000028 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000910 | +0.000556 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000823 | +0.000342 |
| employment_type | salaried | 33473 | 4432 | +0.000119 | +0.000287 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000291 | +0.000111 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000374 | +0.000192 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000504 | +0.000137 |
| aux_account_months | zero | 21000 | 2461 | -0.000600 | +0.000050 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000993 | +0.000004 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001247 | +0.000087 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001335 | +0.000167 |
| employment_type | self_employed | 20987 | 2895 | -0.001516 | +0.000139 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001769 | +0.000256 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002005 | +0.001031 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002138 | +0.000206 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002496 | -0.000400 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.