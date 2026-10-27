# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 14 | 0.5 | 0.42857142857142855 |
| 0.200000 | 1 | 20 | 0.15 | 0.1 |
| 0.323686 | -4 | 22 | 0.045454545454545456 | 0.22727272727272727 |
| 0.400000 | 4 | 14 | 0.35714285714285715 | 0.07142857142857142 |
| 0.500000 | 2 | 23 | 0.08695652173913043 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.000715 | -0.000166 |
| employment_type | variable_income | 15540 | 2180 | +0.000290 | -0.000029 |
| employment_type | self_employed | 20987 | 2895 | +0.000221 | -0.000031 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000102 | -0.000119 |
| employment_type | salaried | 33473 | 4432 | +0.000033 | -0.000041 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000008 | -0.000035 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000049 | -0.000040 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000076 | -0.000026 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000168 | -0.000032 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000291 | -0.000038 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000339 | -0.000035 |
| aux_account_months | zero | 21000 | 2461 | -0.000356 | -0.000020 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000358 | -0.000038 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000586 | -0.000016 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000612 | -0.000059 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001165 | -0.000207 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001612 | -0.000004 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.002093 | -0.000260 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.