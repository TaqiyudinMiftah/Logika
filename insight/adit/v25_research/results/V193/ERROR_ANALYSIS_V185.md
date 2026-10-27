# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 15 | 0.3333333333333333 | 0.4 |
| 0.200000 | 0 | 22 | 0.18181818181818182 | 0.18181818181818182 |
| 0.323686 | 5 | 26 | 0.3076923076923077 | 0.11538461538461539 |
| 0.400000 | -3 | 20 | 0.0 | 0.15 |
| 0.500000 | 0 | 27 | 0.037037037037037035 | 0.037037037037037035 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000542 | -0.000000 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000523 | +0.000008 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000511 | -0.000022 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000492 | +0.000001 |
| aux_account_months | zero | 21000 | 2461 | +0.000442 | -0.000031 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000395 | +0.000005 |
| employment_type | self_employed | 20987 | 2895 | +0.000359 | +0.000001 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000331 | -0.000009 |
| employment_type | variable_income | 15540 | 2180 | +0.000309 | +0.000006 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000231 | -0.000017 |
| employment_type | salaried | 33473 | 4432 | +0.000127 | -0.000004 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000054 | +0.000018 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000000 | -0.000015 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000223 | +0.000015 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000224 | +0.000002 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000718 | +0.000017 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.001372 | -0.000030 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002161 | +0.000099 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.