# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 11 | 0.45454545454545453 | 0.45454545454545453 |
| 0.200000 | 1 | 11 | 0.2727272727272727 | 0.18181818181818182 |
| 0.323686 | -1 | 8 | 0.125 | 0.25 |
| 0.400000 | 3 | 10 | 0.4 | 0.1 |
| 0.500000 | 1 | 10 | 0.2 | 0.1 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.000225 | +0.000065 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000187 | -0.000015 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000092 | -0.000033 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.000057 | -0.000205 |
| aux_account_months | zero | 21000 | 2461 | +0.000054 | -0.000034 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000023 | -0.000037 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000005 | -0.000026 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000000 | +0.000040 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000004 | -0.000103 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000048 | -0.000210 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000070 | -0.000009 |
| employment_type | salaried | 33473 | 4432 | -0.000103 | -0.000027 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000136 | -0.000002 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000167 | -0.000041 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000172 | -0.000036 |
| employment_type | self_employed | 20987 | 2895 | -0.000276 | -0.000124 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000941 | -0.000058 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001933 | -0.000495 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.