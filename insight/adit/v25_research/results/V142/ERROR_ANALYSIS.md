# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -19 | 335 | 0.3611940298507463 | 0.417910447761194 |
| 0.200000 | -7 | 571 | 0.2136602451838879 | 0.22591943957968477 |
| 0.323686 | -39 | 660 | 0.11515151515151516 | 0.17424242424242425 |
| 0.400000 | -10 | 739 | 0.09472259810554803 | 0.10825439783491204 |
| 0.500000 | -12 | 763 | 0.06290956749672345 | 0.07863695937090433 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.010560 | -0.001960 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.003144 | -0.003899 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002616 | -0.002809 |
| employment_type | self_employed | 20987 | 2895 | -0.001258 | -0.001112 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001437 | -0.000199 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001722 | -0.000666 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001953 | -0.000870 |
| aux_account_months | zero | 21000 | 2461 | -0.002377 | -0.000988 |
| employment_type | salaried | 33473 | 4432 | -0.003087 | -0.000528 |
| employment_type | variable_income | 15540 | 2180 | -0.003127 | -0.001190 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.004427 | -0.003504 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.004896 | -0.000786 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.005494 | -0.001302 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.005640 | -0.000854 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.005977 | -0.000814 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.008059 | -0.001507 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.008450 | -0.001136 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.008954 | -0.005226 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.