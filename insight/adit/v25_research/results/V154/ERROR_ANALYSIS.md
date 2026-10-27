# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -4 | 22 | 0.3181818181818182 | 0.5 |
| 0.200000 | 4 | 31 | 0.1935483870967742 | 0.06451612903225806 |
| 0.323686 | -4 | 36 | 0.1388888888888889 | 0.25 |
| 0.400000 | 2 | 31 | 0.22580645161290322 | 0.16129032258064516 |
| 0.500000 | 2 | 34 | 0.058823529411764705 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.000739 | -0.000042 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000731 | -0.000067 |
| employment_type | salaried | 33473 | 4432 | +0.000178 | -0.000067 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000046 | -0.000053 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000057 | -0.000010 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000072 | -0.000272 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000075 | -0.000053 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000160 | -0.000065 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000166 | -0.000061 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000281 | -0.000058 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000328 | -0.000044 |
| aux_account_months | zero | 21000 | 2461 | -0.000388 | -0.000038 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000437 | -0.000218 |
| employment_type | self_employed | 20987 | 2895 | -0.000511 | -0.000051 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000679 | -0.000027 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001311 | -0.000104 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001859 | -0.000336 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.002801 | -0.000421 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.