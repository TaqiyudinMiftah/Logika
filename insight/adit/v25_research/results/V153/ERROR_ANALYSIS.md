# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 14 | 0.42857142857142855 | 0.42857142857142855 |
| 0.200000 | 2 | 24 | 0.16666666666666666 | 0.08333333333333333 |
| 0.323686 | -5 | 27 | 0.037037037037037035 | 0.2222222222222222 |
| 0.400000 | 2 | 18 | 0.2777777777777778 | 0.16666666666666666 |
| 0.500000 | 2 | 25 | 0.08 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.000615 | -0.000194 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000575 | -0.000048 |
| employment_type | variable_income | 15540 | 2180 | +0.000514 | -0.000033 |
| employment_type | salaried | 33473 | 4432 | +0.000033 | -0.000048 |
| employment_type | self_employed | 20987 | 2895 | -0.000069 | -0.000037 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000079 | -0.000038 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000189 | -0.000040 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000278 | -0.000031 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000284 | -0.000045 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000293 | -0.000046 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000366 | -0.000040 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000409 | -0.000151 |
| aux_account_months | zero | 21000 | 2461 | -0.000446 | -0.000024 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000586 | -0.000019 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000612 | -0.000071 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001246 | -0.000234 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001612 | -0.000006 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.002292 | -0.000310 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.