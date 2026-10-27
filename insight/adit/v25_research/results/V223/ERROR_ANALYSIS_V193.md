# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 13 | 251 | 0.3904382470119522 | 0.3386454183266932 |
| 0.200000 | -1 | 380 | 0.2236842105263158 | 0.22631578947368422 |
| 0.323686 | 5 | 440 | 0.1318181818181818 | 0.12045454545454545 |
| 0.400000 | 5 | 480 | 0.0875 | 0.07708333333333334 |
| 0.500000 | 5 | 466 | 0.0815450643776824 | 0.07081545064377683 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.002650 | -0.000084 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002343 | -0.000047 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001732 | -0.000063 |
| aux_account_months | zero | 21000 | 2461 | +0.001496 | -0.000211 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001207 | +0.000052 |
| employment_type | variable_income | 15540 | 2180 | +0.000033 | -0.000351 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000639 | -0.000023 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000751 | -0.000105 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000869 | -0.000088 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000908 | +0.000889 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000966 | -0.000015 |
| baseline_risk_band | under005 | 29938 | 544 | -0.001225 | -0.000397 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001782 | -0.000516 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001788 | +0.000073 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001859 | -0.001014 |
| employment_type | salaried | 33473 | 4432 | -0.001896 | -0.000042 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002349 | +0.000113 |
| employment_type | self_employed | 20987 | 2895 | -0.002434 | +0.000072 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.