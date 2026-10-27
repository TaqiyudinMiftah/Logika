# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 229 | 0.3406113537117904 | 0.34934497816593885 |
| 0.200000 | -2 | 336 | 0.2113095238095238 | 0.21726190476190477 |
| 0.323686 | -12 | 392 | 0.1096938775510204 | 0.14030612244897958 |
| 0.400000 | 4 | 430 | 0.09767441860465116 | 0.08837209302325581 |
| 0.500000 | -1 | 417 | 0.07434052757793765 | 0.07673860911270983 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.006389 | +0.000308 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.004489 | +0.000017 |
| baseline_risk_band | under005 | 29853 | 535 | +0.003841 | +0.002144 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.003272 | +0.001109 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.002217 | +0.000026 |
| employment_type | salaried | 33473 | 4432 | +0.002075 | +0.000090 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.001169 | +0.000317 |
| employment_type | variable_income | 15540 | 2180 | +0.001122 | +0.000406 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000998 | -0.000998 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000660 | +0.000082 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000213 | +0.000110 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000293 | +0.000037 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000483 | +0.000057 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000636 | -0.000090 |
| employment_type | self_employed | 20987 | 2895 | -0.000662 | -0.000020 |
| aux_account_months | zero | 21000 | 2461 | -0.000727 | +0.000347 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000896 | +0.000168 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001089 | +0.000222 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001818 | +0.000214 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001948 | +0.000139 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003787 | -0.000085 |
| orders90_exposure_band | zero | 3428 | 106 | -0.007827 | +0.000321 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.