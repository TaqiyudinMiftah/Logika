# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 25 | 0.32 | 0.32 |
| 0.200000 | -3 | 50 | 0.16 | 0.22 |
| 0.323686 | -1 | 51 | 0.09803921568627451 | 0.11764705882352941 |
| 0.400000 | 0 | 63 | 0.09523809523809523 | 0.09523809523809523 |
| 0.500000 | 0 | 67 | 0.04477611940298507 | 0.04477611940298507 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.001815 | -0.000054 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000849 | -0.000019 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000722 | -0.000047 |
| employment_type | variable_income | 15540 | 2180 | +0.000613 | +0.000005 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000410 | -0.000041 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000299 | -0.000295 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000161 | -0.000051 |
| employment_type | salaried | 33473 | 4432 | +0.000130 | -0.000045 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000029 | -0.000041 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000029 | -0.000051 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000104 | +0.000039 |
| employment_type | self_employed | 20987 | 2895 | -0.000290 | -0.000023 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000482 | -0.000031 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000583 | +0.000018 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000622 | -0.000001 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000633 | +0.000035 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000700 | -0.000040 |
| aux_account_months | zero | 21000 | 2461 | -0.000784 | +0.000040 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000802 | +0.000028 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001078 | -0.000109 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001100 | -0.000319 |
| orders90_exposure_band | zero | 3428 | 106 | -0.003010 | +0.000020 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.