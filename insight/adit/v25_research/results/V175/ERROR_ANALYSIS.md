# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 129 | 0.4186046511627907 | 0.35658914728682173 |
| 0.200000 | 7 | 220 | 0.25 | 0.21818181818181817 |
| 0.323686 | -27 | 274 | 0.07664233576642336 | 0.17518248175182483 |
| 0.400000 | 8 | 320 | 0.10625 | 0.08125 |
| 0.500000 | -4 | 333 | 0.06906906906906907 | 0.08108108108108109 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.004107 | +0.002273 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001957 | +0.000232 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000285 | +0.001151 |
| employment_type | variable_income | 15540 | 2180 | -0.000031 | +0.000312 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000113 | +0.001051 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000218 | +0.000111 |
| employment_type | salaried | 33473 | 4432 | -0.000839 | +0.000052 |
| aux_account_months | zero | 21000 | 2461 | -0.000928 | -0.000074 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000974 | +0.000158 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001089 | +0.000017 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001316 | +0.000015 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001582 | +0.000250 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001957 | +0.000224 |
| employment_type | self_employed | 20987 | 2895 | -0.002018 | +0.000240 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002048 | +0.001145 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002593 | +0.000261 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002834 | +0.000186 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003720 | +0.000210 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.