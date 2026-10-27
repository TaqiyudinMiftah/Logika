# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 86 | 0.4069767441860465 | 0.4186046511627907 |
| 0.200000 | -9 | 147 | 0.17006802721088435 | 0.23129251700680273 |
| 0.323686 | -9 | 162 | 0.09259259259259259 | 0.14814814814814814 |
| 0.400000 | 2 | 202 | 0.1188118811881188 | 0.10891089108910891 |
| 0.500000 | 1 | 185 | 0.07027027027027027 | 0.06486486486486487 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002415 | +0.000310 |
| employment_type | variable_income | 15540 | 2180 | +0.001603 | -0.000004 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000891 | -0.000138 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000063 | +0.000167 |
| employment_type | salaried | 33473 | 4432 | -0.000014 | +0.000070 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000041 | +0.000720 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000072 | +0.000377 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000236 | +0.000090 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000279 | +0.000027 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000479 | +0.000060 |
| aux_account_months | zero | 21000 | 2461 | -0.000521 | +0.000057 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000678 | -0.000033 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000747 | +0.000034 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000783 | +0.000036 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001173 | +0.000067 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001351 | +0.000068 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001554 | +0.000008 |
| employment_type | self_employed | 20987 | 2895 | -0.001700 | +0.000064 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.