# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 16 | 0.5 | 0.5 |
| 0.200000 | 0 | 34 | 0.17647058823529413 | 0.17647058823529413 |
| 0.323686 | 1 | 38 | 0.10526315789473684 | 0.07894736842105263 |
| 0.400000 | 1 | 40 | 0.1 | 0.075 |
| 0.500000 | -2 | 47 | 0.02127659574468085 | 0.06382978723404255 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.002654 | +0.000313 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001782 | -0.000011 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001533 | +0.000057 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001021 | +0.000021 |
| employment_type | variable_income | 15540 | 2180 | +0.000970 | +0.000032 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000804 | +0.000025 |
| employment_type | self_employed | 20987 | 2895 | +0.000732 | +0.000036 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000555 | +0.000086 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000490 | +0.000003 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000322 | +0.000013 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000259 | +0.000014 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000057 | +0.000005 |
| employment_type | salaried | 33473 | 4432 | -0.000037 | -0.000008 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000087 | +0.000010 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000135 | +0.000014 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000176 | +0.000009 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000223 | -0.000009 |
| aux_account_months | zero | 21000 | 2461 | -0.000406 | +0.000004 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000539 | +0.000017 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000711 | +0.000064 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001991 | -0.000057 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002107 | +0.000293 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.