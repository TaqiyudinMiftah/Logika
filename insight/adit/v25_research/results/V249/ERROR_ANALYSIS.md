# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 110 | 0.38181818181818183 | 0.33636363636363636 |
| 0.200000 | 1 | 158 | 0.24050632911392406 | 0.23417721518987342 |
| 0.323686 | -12 | 190 | 0.10526315789473684 | 0.16842105263157894 |
| 0.400000 | -2 | 202 | 0.10396039603960396 | 0.11386138613861387 |
| 0.500000 | 3 | 210 | 0.05238095238095238 | 0.0380952380952381 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.007527 | -0.000256 |
| orders90_exposure_band | zero | 3428 | 106 | +0.001204 | +0.000071 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000461 | -0.000760 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000277 | -0.000460 |
| employment_type | salaried | 33473 | 4432 | +0.000098 | -0.000116 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000017 | -0.000698 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000061 | -0.000197 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000210 | -0.000095 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000268 | -0.000096 |
| employment_type | variable_income | 15540 | 2180 | -0.000361 | -0.000024 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000945 | -0.000110 |
| employment_type | self_employed | 20987 | 2895 | -0.001174 | -0.000183 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001224 | -0.000152 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001238 | -0.000088 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001596 | -0.000022 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001697 | -0.000113 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001881 | -0.000125 |
| aux_account_months | zero | 21000 | 2461 | -0.002140 | -0.000036 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002184 | -0.000137 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002234 | -0.000114 |
| baseline_risk_band | under005 | 29853 | 535 | -0.002729 | -0.001038 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003178 | -0.000058 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.