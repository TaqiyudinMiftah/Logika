# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -6 | 146 | 0.3356164383561644 | 0.3767123287671233 |
| 0.200000 | 27 | 240 | 0.2708333333333333 | 0.15833333333333333 |
| 0.323686 | -12 | 277 | 0.10108303249097472 | 0.1444043321299639 |
| 0.400000 | -4 | 298 | 0.09395973154362416 | 0.10738255033557047 |
| 0.500000 | -12 | 303 | 0.039603960396039604 | 0.07920792079207921 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.009168 | +0.002571 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.004417 | -0.000669 |
| employment_type | variable_income | 15540 | 2180 | +0.003795 | -0.000048 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.002478 | +0.000766 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002469 | -0.000163 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000503 | +0.001458 |
| employment_type | salaried | 33473 | 4432 | +0.000271 | +0.000258 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000133 | +0.000200 |
| aux_account_months | zero | 21000 | 2461 | -0.000194 | +0.000144 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000582 | +0.000148 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000600 | +0.000095 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000719 | +0.000107 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000742 | +0.000372 |
| employment_type | self_employed | 20987 | 2895 | -0.000802 | +0.000127 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000837 | +0.000196 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000903 | +0.000066 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000964 | +0.000145 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001134 | +0.000240 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001976 | +0.000149 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003223 | -0.000429 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003635 | +0.000266 |
| orders90_exposure_band | zero | 3428 | 106 | -0.006623 | +0.000699 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.