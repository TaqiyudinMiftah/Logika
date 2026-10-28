# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 93 | 0.3655913978494624 | 0.3655913978494624 |
| 0.200000 | 6 | 139 | 0.2517985611510791 | 0.20863309352517986 |
| 0.323686 | -8 | 169 | 0.10650887573964497 | 0.15384615384615385 |
| 0.400000 | -5 | 194 | 0.08762886597938144 | 0.1134020618556701 |
| 0.500000 | -1 | 185 | 0.04864864864864865 | 0.05405405405405406 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002988 | -0.000532 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.002881 | +0.000013 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001753 | -0.000596 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000893 | -0.000171 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000572 | -0.000204 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000570 | -0.000106 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000301 | +0.000119 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000228 | -0.000587 |
| employment_type | variable_income | 15540 | 2180 | +0.000156 | -0.000081 |
| employment_type | salaried | 33473 | 4432 | -0.000244 | -0.000098 |
| aux_account_months | zero | 21000 | 2461 | -0.000306 | -0.000142 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000417 | -0.000171 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000484 | -0.000180 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000686 | -0.000172 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000742 | -0.000101 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001158 | -0.000136 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001231 | -0.001096 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001272 | -0.000107 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001274 | +0.000072 |
| employment_type | self_employed | 20987 | 2895 | -0.001506 | -0.000243 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001591 | -0.000199 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003426 | -0.000056 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.