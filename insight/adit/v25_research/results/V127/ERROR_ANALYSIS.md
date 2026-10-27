# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 34 | 0.4117647058823529 | 0.35294117647058826 |
| 0.200000 | 2 | 50 | 0.16 | 0.12 |
| 0.323686 | -7 | 62 | 0.06451612903225806 | 0.1774193548387097 |
| 0.400000 | 1 | 69 | 0.10144927536231885 | 0.08695652173913043 |
| 0.500000 | -1 | 73 | 0.0410958904109589 | 0.0547945205479452 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.001059 | -0.000080 |
| employment_type | variable_income | 15540 | 2180 | +0.000922 | -0.000081 |
| employment_type | self_employed | 20987 | 2895 | +0.000387 | -0.000068 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000359 | -0.000094 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000347 | -0.000066 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000161 | -0.000087 |
| employment_type | salaried | 33473 | 4432 | +0.000078 | -0.000095 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000019 | -0.000367 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000023 | -0.000076 |
| aux_account_months | zero | 21000 | 2461 | -0.000345 | -0.000053 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000423 | -0.000777 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000430 | -0.000061 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000577 | -0.000075 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000646 | -0.000122 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000649 | -0.000105 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001037 | -0.000108 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002324 | -0.000314 |
| baseline_risk_band | under005 | 29853 | 535 | -0.002572 | -0.000735 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.