# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -12 | 320 | 0.38125 | 0.41875 |
| 0.200000 | -5 | 484 | 0.2066115702479339 | 0.21694214876033058 |
| 0.323686 | -26 | 577 | 0.10918544194107452 | 0.15424610051993068 |
| 0.400000 | 6 | 664 | 0.11295180722891567 | 0.10391566265060241 |
| 0.500000 | -15 | 656 | 0.0625 | 0.08536585365853659 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.007044 | -0.001537 |
| employment_type | salaried | 33473 | 4432 | -0.000282 | -0.000224 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000382 | -0.000884 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001599 | -0.000245 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002211 | -0.001633 |
| employment_type | variable_income | 15540 | 2180 | -0.002520 | -0.000437 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002770 | -0.000368 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002799 | -0.000566 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002917 | -0.000197 |
| aux_account_months | zero | 21000 | 2461 | -0.002978 | -0.000200 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003142 | -0.000116 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.003667 | -0.000338 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.003807 | -0.000351 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003827 | -0.000208 |
| baseline_risk_band | under005 | 29853 | 535 | -0.003909 | -0.000072 |
| employment_type | self_employed | 20987 | 2895 | -0.004891 | -0.000216 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005138 | -0.000885 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.006567 | -0.000528 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.