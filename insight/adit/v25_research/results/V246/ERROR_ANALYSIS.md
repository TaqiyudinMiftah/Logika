# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 93 | 0.3548387096774194 | 0.34408602150537637 |
| 0.200000 | 5 | 137 | 0.24817518248175183 | 0.2116788321167883 |
| 0.323686 | -14 | 164 | 0.0975609756097561 | 0.18292682926829268 |
| 0.400000 | -9 | 187 | 0.0748663101604278 | 0.12299465240641712 |
| 0.500000 | 0 | 178 | 0.05056179775280899 | 0.05056179775280899 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.001948 | -0.000193 |
| orders90_exposure_band | zero | 3428 | 106 | +0.001806 | +0.000031 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000955 | -0.000539 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000624 | -0.000077 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000228 | -0.000346 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000249 | -0.000099 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000360 | -0.000107 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000399 | -0.000522 |
| employment_type | salaried | 33473 | 4432 | -0.000674 | -0.000056 |
| employment_type | variable_income | 15540 | 2180 | -0.000814 | -0.000051 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000831 | +0.000011 |
| employment_type | self_employed | 20987 | 2895 | -0.000842 | -0.000189 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.001014 | -0.000111 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001049 | -0.000128 |
| aux_account_months | zero | 21000 | 2461 | -0.001057 | -0.000068 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001405 | -0.000108 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001856 | -0.000096 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001912 | -0.000129 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001956 | -0.000074 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002023 | -0.000071 |
| baseline_risk_band | under005 | 29853 | 535 | -0.002927 | -0.000909 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003843 | -0.000039 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.