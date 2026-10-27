# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -52 | 855 | 0.36140350877192984 | 0.4222222222222222 |
| 0.200000 | -37 | 1284 | 0.20950155763239875 | 0.2383177570093458 |
| 0.323686 | -75 | 1616 | 0.11200495049504951 | 0.15841584158415842 |
| 0.400000 | -28 | 1762 | 0.09704880817253121 | 0.11293984108967083 |
| 0.500000 | -47 | 1880 | 0.06170212765957447 | 0.08670212765957447 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000019 | -0.002004 |
| employment_type | variable_income | 15540 | 2180 | -0.005005 | -0.004286 |
| aux_account_months | zero | 21000 | 2461 | -0.006092 | -0.003014 |
| employment_type | salaried | 33473 | 4432 | -0.006192 | -0.003250 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.006503 | -0.013110 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.006864 | -0.004108 |
| aux_allocation_count | zero | 23732 | 2895 | -0.007985 | -0.003177 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.008203 | -0.002936 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.008463 | -0.003679 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.008741 | -0.004102 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.009241 | -0.004138 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.009621 | -0.003653 |
| employment_type | self_employed | 20987 | 2895 | -0.011345 | -0.004361 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.012020 | -0.004338 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.012208 | -0.004273 |
| baseline_risk_band | under005 | 29853 | 535 | -0.015465 | -0.017949 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.021743 | -0.015018 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.022909 | -0.012819 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.