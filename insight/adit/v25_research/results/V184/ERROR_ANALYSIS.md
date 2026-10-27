# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -8 | 128 | 0.28125 | 0.34375 |
| 0.200000 | 8 | 170 | 0.2529411764705882 | 0.20588235294117646 |
| 0.323686 | -15 | 213 | 0.09859154929577464 | 0.16901408450704225 |
| 0.400000 | -8 | 242 | 0.0743801652892562 | 0.10743801652892562 |
| 0.500000 | -1 | 220 | 0.06363636363636363 | 0.06818181818181818 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.004854 | +0.000078 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001909 | +0.000437 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001744 | +0.000249 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.001512 | +0.000510 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001470 | +0.000962 |
| employment_type | salaried | 33473 | 4432 | +0.001095 | +0.000107 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000368 | +0.000539 |
| aux_account_months | zero | 21000 | 2461 | -0.000079 | +0.000263 |
| employment_type | variable_income | 15540 | 2180 | -0.000120 | +0.000301 |
| employment_type | self_employed | 20987 | 2895 | -0.000206 | +0.000021 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000279 | +0.000114 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000369 | +0.000071 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000666 | +0.000087 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000672 | +0.000174 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000902 | +0.000132 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000937 | +0.000138 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001285 | +0.000026 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001763 | +0.000192 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.