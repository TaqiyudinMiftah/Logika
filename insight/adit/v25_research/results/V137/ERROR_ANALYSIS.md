# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 3 | 95 | 0.4 | 0.3684210526315789 |
| 0.200000 | 8 | 135 | 0.2814814814814815 | 0.2222222222222222 |
| 0.323686 | -14 | 167 | 0.0658682634730539 | 0.1497005988023952 |
| 0.400000 | -1 | 189 | 0.07407407407407407 | 0.07936507936507936 |
| 0.500000 | 4 | 195 | 0.08205128205128205 | 0.06153846153846154 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.001215 | -0.000751 |
| employment_type | salaried | 33473 | 4432 | +0.000689 | -0.000099 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000364 | +0.000021 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000164 | -0.001582 |
| aux_account_months | zero | 21000 | 2461 | -0.000212 | -0.000074 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000223 | +0.000032 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000449 | -0.000064 |
| employment_type | self_employed | 20987 | 2895 | -0.000550 | -0.000125 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000667 | -0.000119 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000839 | -0.000088 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001006 | -0.000200 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001242 | -0.000175 |
| employment_type | variable_income | 15540 | 2180 | -0.001305 | -0.000258 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001380 | -0.000203 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001801 | -0.000681 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002062 | +0.000175 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.002488 | -0.001774 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.004645 | -0.000385 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.