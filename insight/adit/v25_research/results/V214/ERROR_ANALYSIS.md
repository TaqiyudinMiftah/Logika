# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 114 | 0.32456140350877194 | 0.3333333333333333 |
| 0.200000 | 4 | 166 | 0.23493975903614459 | 0.21084337349397592 |
| 0.323686 | -21 | 207 | 0.07729468599033816 | 0.178743961352657 |
| 0.400000 | -5 | 213 | 0.07511737089201878 | 0.09859154929577464 |
| 0.500000 | -5 | 217 | 0.041474654377880185 | 0.06451612903225806 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.007258 | +0.000100 |
| employment_type | variable_income | 15540 | 2180 | +0.001236 | -0.000192 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001031 | -0.000617 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000891 | -0.000572 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000825 | +0.000042 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000736 | +0.000147 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000527 | -0.000027 |
| employment_type | self_employed | 20987 | 2895 | +0.000197 | +0.000008 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000000 | +0.002357 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000415 | +0.000039 |
| aux_account_months | zero | 21000 | 2461 | -0.000511 | +0.000003 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000567 | +0.000009 |
| employment_type | salaried | 33473 | 4432 | -0.000730 | +0.000158 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000784 | -0.000145 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.001239 | +0.000044 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001374 | +0.001347 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001442 | +0.000490 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001695 | +0.000045 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001746 | -0.000146 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001771 | +0.000070 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002300 | +0.000161 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002647 | +0.000015 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.