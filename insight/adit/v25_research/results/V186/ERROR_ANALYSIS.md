# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 13 | 0.38461538461538464 | 0.5384615384615384 |
| 0.200000 | 1 | 17 | 0.29411764705882354 | 0.23529411764705882 |
| 0.323686 | -1 | 32 | 0.09375 | 0.125 |
| 0.400000 | 0 | 48 | 0.10416666666666667 | 0.10416666666666667 |
| 0.500000 | -2 | 56 | 0.03571428571428571 | 0.07142857142857142 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.001494 | +0.000105 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.001305 | -0.000004 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001080 | -0.000029 |
| employment_type | variable_income | 15540 | 2180 | +0.000950 | -0.000032 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000887 | +0.000019 |
| employment_type | self_employed | 20987 | 2895 | +0.000497 | -0.000020 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000318 | -0.000019 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000299 | -0.000004 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000092 | -0.000033 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000057 | -0.000044 |
| employment_type | salaried | 33473 | 4432 | +0.000043 | +0.000031 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000011 | +0.000021 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000051 | +0.000023 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000126 | +0.000024 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000145 | +0.000040 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000273 | +0.000010 |
| aux_account_months | zero | 21000 | 2461 | -0.000324 | -0.000047 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000623 | +0.000003 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000728 | -0.000021 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000742 | +0.000003 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001205 | +0.000097 |
| orders90_exposure_band | zero | 3428 | 106 | -0.001806 | -0.000099 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.