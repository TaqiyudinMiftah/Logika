# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -3 | 313 | 0.3993610223642173 | 0.40894568690095845 |
| 0.200000 | -3 | 471 | 0.20382165605095542 | 0.21019108280254778 |
| 0.323686 | -26 | 564 | 0.09219858156028368 | 0.13829787234042554 |
| 0.400000 | 8 | 621 | 0.1143317230273752 | 0.10144927536231885 |
| 0.500000 | -9 | 596 | 0.06879194630872483 | 0.08389261744966443 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.007502 | -0.001227 |
| employment_type | salaried | 33473 | 4432 | -0.000159 | -0.000015 |
| employment_type | variable_income | 15540 | 2180 | -0.000938 | -0.000456 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001578 | -0.000315 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001628 | -0.000288 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001729 | -0.000116 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001876 | -0.002084 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002364 | -0.001309 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002427 | +0.000361 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002951 | -0.000271 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.003114 | -0.000270 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.003460 | -0.000369 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.003485 | -0.000224 |
| aux_account_months | zero | 21000 | 2461 | -0.003600 | -0.000078 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003651 | -0.000021 |
| employment_type | self_employed | 20987 | 2895 | -0.003826 | -0.000364 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003977 | -0.000122 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.005551 | -0.000320 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.