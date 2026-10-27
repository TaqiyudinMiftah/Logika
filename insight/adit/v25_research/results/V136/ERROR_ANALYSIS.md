# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 120 | 0.4166666666666667 | 0.4083333333333333 |
| 0.200000 | 4 | 179 | 0.26256983240223464 | 0.24022346368715083 |
| 0.323686 | -18 | 216 | 0.08796296296296297 | 0.1712962962962963 |
| 0.400000 | 4 | 237 | 0.08438818565400844 | 0.06751054852320675 |
| 0.500000 | 5 | 248 | 0.0846774193548387 | 0.06451612903225806 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.001788 | -0.001021 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001637 | -0.001949 |
| employment_type | salaried | 33473 | 4432 | +0.000711 | -0.000131 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000245 | -0.000032 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000219 | +0.000038 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000137 | -0.000086 |
| aux_account_months | zero | 21000 | 2461 | -0.000364 | -0.000087 |
| employment_type | self_employed | 20987 | 2895 | -0.000661 | -0.000179 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000839 | -0.002286 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001163 | -0.000140 |
| employment_type | variable_income | 15540 | 2180 | -0.001205 | -0.000350 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001353 | -0.000121 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001593 | -0.001069 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001778 | -0.000239 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001937 | -0.000273 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002037 | -0.000264 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002884 | +0.000101 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.005735 | -0.000512 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.