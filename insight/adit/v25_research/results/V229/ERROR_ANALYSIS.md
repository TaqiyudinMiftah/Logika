# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 23 | 0.5652173913043478 | 0.34782608695652173 |
| 0.200000 | 1 | 47 | 0.2553191489361702 | 0.23404255319148937 |
| 0.323686 | -3 | 40 | 0.05 | 0.125 |
| 0.400000 | -2 | 54 | 0.09259259259259259 | 0.12962962962962962 |
| 0.500000 | 2 | 54 | 0.05555555555555555 | 0.018518518518518517 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.001204 | +0.000043 |
| employment_type | self_employed | 20987 | 2895 | +0.000456 | +0.000024 |
| employment_type | salaried | 33473 | 4432 | +0.000426 | +0.000055 |
| employment_type | variable_income | 15540 | 2180 | +0.000402 | +0.000039 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000386 | +0.000072 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000321 | -0.000041 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000285 | +0.000352 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000278 | +0.000355 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000189 | -0.000120 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000126 | +0.000020 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000005 | +0.000032 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000030 | +0.000161 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000096 | +0.000140 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000104 | +0.000075 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000142 | +0.000016 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000160 | +0.000020 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000233 | +0.000097 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000370 | +0.000080 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000388 | +0.000054 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000433 | +0.000044 |
| aux_account_months | zero | 21000 | 2461 | -0.000475 | +0.000112 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001104 | +0.000023 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.