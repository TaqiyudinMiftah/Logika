# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -7 | 285 | 0.34385964912280703 | 0.3684210526315789 |
| 0.200000 | -13 | 482 | 0.2053941908713693 | 0.23236514522821577 |
| 0.323686 | -32 | 599 | 0.1018363939899833 | 0.15525876460767946 |
| 0.400000 | -21 | 644 | 0.09006211180124224 | 0.12267080745341614 |
| 0.500000 | -20 | 715 | 0.06433566433566433 | 0.09230769230769231 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002161 | -0.000815 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000597 | -0.005427 |
| employment_type | salaried | 33473 | 4432 | -0.001535 | -0.001741 |
| employment_type | variable_income | 15540 | 2180 | -0.001678 | -0.001732 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002023 | -0.001525 |
| employment_type | self_employed | 20987 | 2895 | -0.002267 | -0.001454 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002481 | -0.001635 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002526 | -0.001750 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002825 | -0.001760 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.003357 | -0.001182 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003592 | -0.002290 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003742 | -0.001509 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003758 | -0.001730 |
| aux_account_months | zero | 21000 | 2461 | -0.004542 | -0.001381 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.005323 | -0.002370 |
| baseline_risk_band | under005 | 29853 | 535 | -0.007975 | -0.007445 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.015201 | -0.011267 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.019917 | -0.008965 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.