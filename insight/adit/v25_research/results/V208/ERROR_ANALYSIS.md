# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -13 | 370 | 0.34324324324324323 | 0.3783783783783784 |
| 0.200000 | 3 | 574 | 0.23344947735191637 | 0.22822299651567945 |
| 0.323686 | -11 | 713 | 0.11640953716690042 | 0.13183730715287517 |
| 0.400000 | -6 | 810 | 0.09259259259259259 | 0.1 |
| 0.500000 | -19 | 839 | 0.06436233611442194 | 0.08700834326579261 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.003294 | -0.001030 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002822 | -0.005969 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001835 | -0.001740 |
| orders90_exposure_band | zero | 3428 | 106 | +0.001795 | -0.000355 |
| employment_type | salaried | 33473 | 4432 | +0.000643 | -0.000962 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000213 | -0.001239 |
| employment_type | variable_income | 15540 | 2180 | -0.000821 | -0.000229 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000972 | -0.000850 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001268 | -0.000007 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001308 | -0.003119 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001612 | -0.000907 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001631 | -0.000892 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001649 | -0.001028 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002740 | -0.001326 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002956 | -0.003043 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002994 | -0.000524 |
| employment_type | self_employed | 20987 | 2895 | -0.003134 | -0.000921 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003233 | -0.000282 |
| aux_account_months | zero | 21000 | 2461 | -0.005330 | -0.000127 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.005961 | -0.000351 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.009572 | -0.001224 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.011140 | -0.003633 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.