# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 15 | 0.4 | 0.5333333333333333 |
| 0.200000 | 1 | 47 | 0.2553191489361702 | 0.23404255319148937 |
| 0.323686 | 0 | 58 | 0.1206896551724138 | 0.1206896551724138 |
| 0.400000 | -2 | 82 | 0.08536585365853659 | 0.10975609756097561 |
| 0.500000 | -5 | 101 | 0.04950495049504951 | 0.09900990099009901 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001782 | +0.000073 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001589 | +0.000247 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000785 | -0.000102 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000636 | +0.000024 |
| employment_type | salaried | 33473 | 4432 | +0.000533 | +0.000015 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000462 | -0.000015 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000374 | +0.000001 |
| employment_type | variable_income | 15540 | 2180 | +0.000077 | +0.000012 |
| employment_type | self_employed | 20987 | 2895 | +0.000069 | -0.000076 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000034 | -0.000069 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000057 | -0.000015 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000100 | -0.000024 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000134 | -0.000091 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000154 | +0.000106 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000304 | -0.000016 |
| aux_account_months | zero | 21000 | 2461 | -0.000543 | -0.000016 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000800 | -0.000036 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000881 | -0.000403 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.