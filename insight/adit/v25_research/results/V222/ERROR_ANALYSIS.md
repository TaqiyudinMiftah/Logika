# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 51 | 0.35294117647058826 | 0.37254901960784315 |
| 0.200000 | 2 | 81 | 0.19753086419753085 | 0.1728395061728395 |
| 0.323686 | -7 | 98 | 0.12244897959183673 | 0.19387755102040816 |
| 0.400000 | 2 | 115 | 0.1391304347826087 | 0.12173913043478261 |
| 0.500000 | -2 | 123 | 0.032520325203252036 | 0.04878048780487805 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002408 | -0.000017 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001054 | -0.000750 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000838 | -0.000136 |
| employment_type | variable_income | 15540 | 2180 | +0.000689 | -0.000014 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000382 | +0.000475 |
| employment_type | self_employed | 20987 | 2895 | +0.000373 | -0.000087 |
| aux_account_months | zero | 21000 | 2461 | +0.000263 | +0.000064 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000210 | +0.000021 |
| employment_type | salaried | 33473 | 4432 | +0.000061 | -0.000055 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000075 | -0.000021 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000124 | +0.000057 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000128 | -0.000085 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000233 | -0.000057 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000258 | -0.000085 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000577 | -0.000282 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001286 | -0.000017 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001947 | -0.000145 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002319 | -0.000103 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.