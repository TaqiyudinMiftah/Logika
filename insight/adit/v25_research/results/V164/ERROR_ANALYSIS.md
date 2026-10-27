# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -37 | 832 | 0.3605769230769231 | 0.4050480769230769 |
| 0.200000 | -29 | 1239 | 0.2066182405165456 | 0.23002421307506055 |
| 0.323686 | -64 | 1574 | 0.11245235069885642 | 0.1531130876747141 |
| 0.400000 | -40 | 1734 | 0.09630911188004614 | 0.11937716262975778 |
| 0.500000 | -36 | 1857 | 0.0630048465266559 | 0.08239095315024232 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | -0.003514 | -0.003027 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.005269 | -0.003250 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.006305 | -0.012460 |
| employment_type | salaried | 33473 | 4432 | -0.006374 | -0.003414 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.006524 | -0.003504 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.007510 | -0.003347 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.008104 | -0.003295 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.008569 | -0.002911 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.008644 | -0.003343 |
| employment_type | self_employed | 20987 | 2895 | -0.008773 | -0.003156 |
| aux_allocation_count | zero | 23732 | 2895 | -0.009653 | -0.003245 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.010036 | -0.003418 |
| aux_account_months | zero | 21000 | 2461 | -0.011053 | -0.003053 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.015148 | -0.003453 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.017916 | -0.004201 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.021142 | -0.010661 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.023330 | -0.013325 |
| baseline_risk_band | under005 | 29853 | 535 | -0.027751 | -0.010556 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.