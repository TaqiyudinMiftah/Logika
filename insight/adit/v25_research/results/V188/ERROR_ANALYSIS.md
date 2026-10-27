# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 40 | 0.3 | 0.325 |
| 0.200000 | 6 | 45 | 0.28888888888888886 | 0.15555555555555556 |
| 0.323686 | -5 | 48 | 0.08333333333333333 | 0.1875 |
| 0.400000 | 2 | 65 | 0.1076923076923077 | 0.07692307692307693 |
| 0.500000 | -3 | 60 | 0.03333333333333333 | 0.08333333333333333 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001930 | +0.000289 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001355 | +0.000021 |
| employment_type | salaried | 33473 | 4432 | +0.000850 | +0.000024 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000553 | +0.000026 |
| aux_account_months | zero | 21000 | 2461 | +0.000539 | +0.000080 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000455 | +0.000109 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000403 | +0.000054 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000359 | +0.000032 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000315 | +0.000044 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000311 | +0.000044 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000273 | +0.000048 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000073 | +0.000028 |
| employment_type | variable_income | 15540 | 2180 | +0.000029 | +0.000070 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000057 | +0.000025 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000471 | +0.000184 |
| orders90_exposure_band | zero | 3428 | 106 | -0.000602 | -0.000006 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000602 | +0.000019 |
| employment_type | self_employed | 20987 | 2895 | -0.000801 | +0.000025 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000984 | -0.000020 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000989 | +0.000238 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001138 | +0.000074 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001231 | -0.000038 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.