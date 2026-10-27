# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 20 | 0.5 | 0.45 |
| 0.200000 | -4 | 45 | 0.17777777777777778 | 0.26666666666666666 |
| 0.323686 | -4 | 42 | 0.07142857142857142 | 0.16666666666666666 |
| 0.400000 | 4 | 55 | 0.18181818181818182 | 0.10909090909090909 |
| 0.500000 | 1 | 54 | 0.05555555555555555 | 0.037037037037037035 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.001674 | +0.000253 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001098 | +0.000371 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001080 | -0.000042 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000778 | +0.000023 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000554 | +0.000042 |
| aux_account_months | zero | 21000 | 2461 | +0.000464 | +0.000108 |
| employment_type | self_employed | 20987 | 2895 | +0.000428 | -0.000010 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000391 | +0.000075 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000358 | +0.000007 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000346 | +0.000104 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000156 | -0.000034 |
| employment_type | salaried | 33473 | 4432 | +0.000111 | +0.000034 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000110 | +0.000007 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000050 | -0.000006 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000073 | -0.000004 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000110 | +0.000052 |
| employment_type | variable_income | 15540 | 2180 | -0.000117 | +0.000045 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000207 | +0.000015 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000603 | -0.000012 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000692 | +0.000065 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001290 | +0.000130 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002101 | +0.000003 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.