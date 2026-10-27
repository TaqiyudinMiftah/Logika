# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -5 | 162 | 0.29012345679012347 | 0.32098765432098764 |
| 0.200000 | 5 | 219 | 0.2328767123287671 | 0.2100456621004566 |
| 0.323686 | -17 | 261 | 0.11494252873563218 | 0.18007662835249041 |
| 0.400000 | 2 | 296 | 0.0945945945945946 | 0.08783783783783784 |
| 0.500000 | -7 | 279 | 0.06451612903225806 | 0.08960573476702509 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.004576 | +0.000059 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.004133 | +0.000369 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003090 | +0.000335 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001658 | +0.000738 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001105 | -0.000839 |
| employment_type | salaried | 33473 | 4432 | +0.000790 | +0.000006 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000765 | -0.000158 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000568 | -0.000066 |
| employment_type | variable_income | 15540 | 2180 | +0.000459 | +0.000194 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000031 | -0.000085 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000320 | +0.000023 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000506 | +0.000007 |
| aux_account_months | zero | 21000 | 2461 | -0.000608 | +0.000150 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000703 | -0.000005 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001044 | +0.000071 |
| employment_type | self_employed | 20987 | 2895 | -0.001103 | -0.000030 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001375 | +0.000079 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001394 | -0.000116 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001618 | +0.000181 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001732 | +0.000058 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001804 | +0.000027 |
| orders90_exposure_band | zero | 3428 | 106 | -0.001806 | +0.000088 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.