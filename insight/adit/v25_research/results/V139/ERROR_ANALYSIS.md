# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 394 | 0.39086294416243655 | 0.3730964467005076 |
| 0.200000 | 15 | 611 | 0.23076923076923078 | 0.20621931260229132 |
| 0.323686 | -48 | 751 | 0.10918774966711052 | 0.17310252996005326 |
| 0.400000 | -9 | 861 | 0.09872241579558652 | 0.1091753774680604 |
| 0.500000 | -17 | 872 | 0.05619266055045872 | 0.07568807339449542 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.006378 | -0.002589 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002789 | -0.005303 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001100 | -0.003966 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001374 | -0.004677 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001499 | -0.000498 |
| employment_type | salaried | 33473 | 4432 | -0.002155 | -0.000679 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002198 | -0.000930 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002918 | -0.001132 |
| aux_account_months | zero | 21000 | 2461 | -0.003452 | -0.001131 |
| employment_type | self_employed | 20987 | 2895 | -0.003732 | -0.001463 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.005047 | -0.001436 |
| employment_type | variable_income | 15540 | 2180 | -0.005363 | -0.001429 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.007075 | -0.001065 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.007215 | -0.000999 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.007394 | -0.001017 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.007855 | -0.005024 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.009409 | -0.001272 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.010556 | -0.001648 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.