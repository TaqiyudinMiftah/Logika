# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 83 | 0.3855421686746988 | 0.39759036144578314 |
| 0.200000 | 13 | 131 | 0.3053435114503817 | 0.20610687022900764 |
| 0.323686 | -14 | 154 | 0.08441558441558442 | 0.17532467532467533 |
| 0.400000 | -1 | 174 | 0.07471264367816093 | 0.08045977011494253 |
| 0.500000 | 0 | 176 | 0.06818181818181818 | 0.06818181818181818 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.005089 | -0.000747 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.003462 | -0.000643 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001327 | -0.000373 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000765 | +0.000533 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000316 | +0.000092 |
| employment_type | salaried | 33473 | 4432 | +0.000255 | +0.000010 |
| aux_account_months | zero | 21000 | 2461 | +0.000111 | -0.000017 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000048 | +0.000044 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000803 | +0.000006 |
| employment_type | variable_income | 15540 | 2180 | -0.000973 | -0.000184 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.001099 | -0.001005 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001145 | +0.000151 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001181 | -0.000072 |
| employment_type | self_employed | 20987 | 2895 | -0.001199 | -0.000043 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001646 | -0.000124 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001669 | -0.000106 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001815 | -0.000017 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003130 | -0.000280 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.