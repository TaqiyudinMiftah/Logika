# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 43 | 0.4883720930232558 | 0.37209302325581395 |
| 0.200000 | 3 | 81 | 0.25925925925925924 | 0.2222222222222222 |
| 0.323686 | -7 | 95 | 0.08421052631578947 | 0.15789473684210525 |
| 0.400000 | -2 | 118 | 0.11016949152542373 | 0.1271186440677966 |
| 0.500000 | 3 | 108 | 0.06481481481481481 | 0.037037037037037035 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002180 | -0.000189 |
| employment_type | variable_income | 15540 | 2180 | +0.001465 | -0.000010 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001279 | +0.000296 |
| employment_type | salaried | 33473 | 4432 | +0.000792 | -0.000001 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000225 | -0.000011 |
| aux_account_months | zero | 21000 | 2461 | +0.000093 | +0.000067 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000050 | +0.000006 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000107 | +0.000085 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000148 | -0.000016 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000242 | +0.000026 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000339 | +0.000029 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000796 | +0.000097 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000809 | +0.000018 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000819 | -0.000283 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000850 | +0.000280 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000923 | -0.000050 |
| employment_type | self_employed | 20987 | 2895 | -0.001189 | +0.000026 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001757 | -0.000158 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.