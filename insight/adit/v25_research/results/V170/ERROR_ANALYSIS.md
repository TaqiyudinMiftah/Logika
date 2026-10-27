# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 124 | 0.3790322580645161 | 0.3467741935483871 |
| 0.200000 | 7 | 219 | 0.2100456621004566 | 0.1780821917808219 |
| 0.323686 | -25 | 287 | 0.10801393728222997 | 0.1951219512195122 |
| 0.400000 | 2 | 328 | 0.10670731707317073 | 0.10060975609756098 |
| 0.500000 | 2 | 348 | 0.07183908045977011 | 0.06609195402298851 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001952 | +0.001352 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000836 | +0.001333 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000604 | +0.000379 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000437 | +0.001544 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000457 | -0.000652 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000520 | +0.000230 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000607 | +0.000273 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000756 | +0.000200 |
| employment_type | salaried | 33473 | 4432 | -0.000756 | +0.000046 |
| employment_type | self_employed | 20987 | 2895 | -0.001047 | +0.000401 |
| employment_type | variable_income | 15540 | 2180 | -0.001250 | +0.000174 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001497 | +0.000396 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001564 | +0.000030 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001878 | +0.000186 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002148 | +0.000046 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002309 | -0.000097 |
| aux_account_months | zero | 21000 | 2461 | -0.002431 | -0.000021 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003766 | -0.000129 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.