# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 12 | 240 | 0.39166666666666666 | 0.3416666666666667 |
| 0.200000 | -6 | 363 | 0.2231404958677686 | 0.2396694214876033 |
| 0.323686 | 12 | 414 | 0.14009661835748793 | 0.1111111111111111 |
| 0.400000 | 9 | 446 | 0.09192825112107623 | 0.07174887892376682 |
| 0.500000 | 6 | 436 | 0.07568807339449542 | 0.06192660550458716 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003992 | +0.000027 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003970 | +0.000018 |
| aux_account_months | zero | 21000 | 2461 | +0.002280 | -0.000152 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001226 | -0.000176 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.001223 | +0.000718 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001116 | +0.000053 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000560 | +0.000012 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000272 | -0.000068 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000409 | -0.000039 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000436 | +0.000001 |
| employment_type | self_employed | 20987 | 2895 | -0.000444 | +0.000029 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000863 | +0.000245 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001394 | +0.000452 |
| employment_type | variable_income | 15540 | 2180 | -0.001540 | -0.000282 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001979 | -0.000002 |
| employment_type | salaried | 33473 | 4432 | -0.002389 | +0.000007 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003716 | -0.000287 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.004080 | -0.001337 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.