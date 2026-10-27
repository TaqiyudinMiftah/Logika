# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -3 | 36 | 0.3333333333333333 | 0.4166666666666667 |
| 0.200000 | -1 | 70 | 0.21428571428571427 | 0.22857142857142856 |
| 0.323686 | -7 | 69 | 0.11594202898550725 | 0.21739130434782608 |
| 0.400000 | -1 | 87 | 0.11494252873563218 | 0.12643678160919541 |
| 0.500000 | -2 | 86 | 0.011627906976744186 | 0.03488372093023256 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002408 | -0.000224 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001004 | -0.000150 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000375 | +0.000159 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000208 | -0.000025 |
| employment_type | salaried | 33473 | 4432 | +0.000050 | -0.000018 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000043 | -0.000026 |
| employment_type | variable_income | 15540 | 2180 | +0.000025 | +0.000005 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000265 | -0.000030 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000304 | -0.000042 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000316 | -0.000030 |
| employment_type | self_employed | 20987 | 2895 | -0.000318 | -0.000019 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000434 | +0.000049 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000459 | +0.000026 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000580 | +0.000030 |
| aux_account_months | zero | 21000 | 2461 | -0.000665 | +0.000074 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000736 | -0.000024 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000745 | +0.000096 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000767 | -0.000028 |
| orders90_exposure_band | zero | 3428 | 106 | -0.000903 | -0.000131 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001048 | -0.000193 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001142 | +0.000010 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001401 | +0.000007 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.