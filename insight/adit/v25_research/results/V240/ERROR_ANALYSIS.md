# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 176 | 0.3522727272727273 | 0.36363636363636365 |
| 0.200000 | 11 | 275 | 0.24 | 0.2 |
| 0.323686 | -22 | 340 | 0.10882352941176471 | 0.17352941176470588 |
| 0.400000 | -10 | 360 | 0.09166666666666666 | 0.11944444444444445 |
| 0.500000 | -8 | 359 | 0.04456824512534819 | 0.06685236768802229 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.006145 | -0.000177 |
| baseline_risk_band | under005 | 29853 | 535 | +0.002736 | +0.002136 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.002179 | +0.000184 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001991 | +0.000296 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000900 | +0.000750 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000635 | +0.000184 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.000527 | +0.000039 |
| employment_type | variable_income | 15540 | 2180 | +0.000479 | -0.000101 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000165 | +0.000020 |
| employment_type | self_employed | 20987 | 2895 | +0.000031 | -0.000037 |
| aux_account_months | zero | 21000 | 2461 | -0.000316 | +0.000014 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000573 | +0.000151 |
| employment_type | salaried | 33473 | 4432 | -0.000575 | +0.000223 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000752 | +0.000006 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000756 | +0.000247 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.001270 | +0.000124 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001496 | +0.000081 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001905 | -0.000003 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002054 | +0.000085 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002744 | -0.000191 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002800 | +0.000040 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.003378 | -0.000146 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.