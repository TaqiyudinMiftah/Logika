# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -3 | 28 | 0.35714285714285715 | 0.4642857142857143 |
| 0.200000 | -5 | 48 | 0.20833333333333334 | 0.3125 |
| 0.323686 | -2 | 55 | 0.09090909090909091 | 0.12727272727272726 |
| 0.400000 | 1 | 41 | 0.04878048780487805 | 0.024390243902439025 |
| 0.500000 | 1 | 61 | 0.09836065573770492 | 0.08196721311475409 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.000978 | -0.000125 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000833 | +0.000004 |
| employment_type | self_employed | 20987 | 2895 | +0.000373 | -0.000062 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000346 | +0.000075 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000230 | +0.000011 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000135 | -0.000025 |
| aux_account_months | zero | 21000 | 2461 | +0.000043 | -0.000002 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000015 | -0.000004 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000009 | +0.000010 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000011 | +0.000347 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000067 | -0.000006 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000115 | +0.000002 |
| employment_type | salaried | 33473 | 4432 | -0.000145 | +0.000015 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000195 | -0.000006 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000377 | +0.000031 |
| employment_type | variable_income | 15540 | 2180 | -0.000526 | +0.000038 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001991 | -0.000008 |
| baseline_risk_band | under005 | 29965 | 546 | -0.002061 | -0.000272 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.