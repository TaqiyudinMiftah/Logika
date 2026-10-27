# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 254 | 0.3700787401574803 | 0.3464566929133858 |
| 0.200000 | 12 | 387 | 0.23772609819121446 | 0.20671834625322996 |
| 0.323686 | 13 | 479 | 0.1336116910229645 | 0.10647181628392484 |
| 0.400000 | -1 | 518 | 0.0888030888030888 | 0.09073359073359073 |
| 0.500000 | -4 | 502 | 0.07171314741035857 | 0.0796812749003984 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003883 | -0.000095 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003842 | -0.000059 |
| baseline_risk_band | under005 | 29965 | 546 | +0.001903 | +0.000683 |
| aux_account_months | zero | 21000 | 2461 | +0.001525 | -0.000261 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.001254 | +0.000064 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001134 | -0.000264 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.000778 | -0.000228 |
| employment_type | self_employed | 20987 | 2895 | +0.000233 | +0.000011 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000177 | +0.000124 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000028 | +0.000126 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000361 | +0.000030 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000666 | -0.000049 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000735 | +0.000070 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.001002 | +0.000478 |
| employment_type | variable_income | 15540 | 2180 | -0.001472 | -0.000369 |
| employment_type | salaried | 33473 | 4432 | -0.002743 | +0.000116 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004732 | +0.000331 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005251 | -0.000084 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.