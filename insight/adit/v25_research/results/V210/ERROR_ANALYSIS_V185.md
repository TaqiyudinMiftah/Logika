# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 24 | 0.4166666666666667 | 0.4166666666666667 |
| 0.200000 | 0 | 37 | 0.21621621621621623 | 0.21621621621621623 |
| 0.323686 | 2 | 54 | 0.14814814814814814 | 0.1111111111111111 |
| 0.400000 | 1 | 41 | 0.04878048780487805 | 0.024390243902439025 |
| 0.500000 | -2 | 46 | 0.021739130434782608 | 0.06521739130434782 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.001281 | +0.000132 |
| aux_account_months | zero | 21000 | 2461 | +0.000831 | +0.000012 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000800 | +0.000046 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000779 | +0.000035 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000600 | +0.000049 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000566 | +0.000263 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000423 | +0.000023 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000408 | +0.000467 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000375 | +0.000023 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000373 | +0.000018 |
| employment_type | self_employed | 20987 | 2895 | +0.000345 | +0.000037 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000344 | +0.000041 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000220 | +0.000184 |
| employment_type | salaried | 33473 | 4432 | +0.000092 | +0.000054 |
| employment_type | variable_income | 15540 | 2180 | -0.000130 | +0.000006 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000414 | +0.000065 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000731 | +0.000106 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002427 | +0.000044 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.