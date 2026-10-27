# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 452 | 0.37610619469026546 | 0.3805309734513274 |
| 0.200000 | -32 | 752 | 0.19946808510638298 | 0.24202127659574468 |
| 0.323686 | -53 | 921 | 0.11509229098805646 | 0.17263843648208468 |
| 0.400000 | -28 | 1025 | 0.08878048780487804 | 0.11609756097560976 |
| 0.500000 | -22 | 1035 | 0.06086956521739131 | 0.0821256038647343 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.008227 | -0.005344 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002010 | -0.003538 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000596 | -0.007041 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001469 | -0.000812 |
| employment_type | self_employed | 20987 | 2895 | -0.002751 | -0.001853 |
| employment_type | salaried | 33473 | 4432 | -0.003801 | -0.001216 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003888 | -0.001501 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.004793 | -0.001749 |
| employment_type | variable_income | 15540 | 2180 | -0.005065 | -0.002065 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.005568 | -0.002082 |
| aux_account_months | zero | 21000 | 2461 | -0.005854 | -0.001762 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.006056 | -0.001504 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.006274 | -0.001551 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.006636 | -0.001568 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.007976 | -0.001699 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.008010 | -0.006523 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.009460 | -0.002396 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.014659 | -0.007847 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.