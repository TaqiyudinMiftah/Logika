# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 11 | 452 | 0.3893805309734513 | 0.36504424778761063 |
| 0.200000 | 10 | 730 | 0.22876712328767124 | 0.21506849315068494 |
| 0.323686 | -48 | 885 | 0.10847457627118644 | 0.16271186440677965 |
| 0.400000 | -6 | 998 | 0.09519038076152304 | 0.10120240480961924 |
| 0.500000 | -14 | 1030 | 0.06019417475728155 | 0.07378640776699029 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.009482 | -0.002614 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000094 | -0.002322 |
| employment_type | self_employed | 20987 | 2895 | -0.001891 | -0.001669 |
| employment_type | salaried | 33473 | 4432 | -0.001917 | -0.000913 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.002066 | -0.006125 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002557 | -0.000854 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003209 | -0.001850 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003512 | -0.001814 |
| aux_account_months | zero | 21000 | 2461 | -0.004754 | -0.001394 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.006420 | -0.001811 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.006528 | -0.005650 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.006754 | -0.001284 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.007083 | -0.001119 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.007593 | -0.001059 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.007758 | -0.001064 |
| employment_type | variable_income | 15540 | 2180 | -0.008702 | -0.001745 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.009817 | -0.001493 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.011441 | -0.005772 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.