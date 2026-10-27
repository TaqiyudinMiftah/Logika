# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 13 | 240 | 0.3958333333333333 | 0.3416666666666667 |
| 0.200000 | -6 | 368 | 0.22554347826086957 | 0.2418478260869565 |
| 0.323686 | 7 | 413 | 0.13317191283292978 | 0.1162227602905569 |
| 0.400000 | 12 | 453 | 0.09713024282560706 | 0.0706401766004415 |
| 0.500000 | 6 | 429 | 0.07459207459207459 | 0.06060606060606061 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003761 | +0.000044 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003459 | +0.000041 |
| aux_account_months | zero | 21000 | 2461 | +0.001838 | -0.000121 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001451 | -0.000178 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001339 | +0.000037 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.001323 | +0.000725 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000229 | +0.000021 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000047 | +0.000237 |
| employment_type | self_employed | 20987 | 2895 | -0.000803 | +0.000028 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000814 | -0.000068 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000901 | -0.000040 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000959 | -0.000006 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001555 | -0.000386 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001784 | +0.000433 |
| employment_type | variable_income | 15540 | 2180 | -0.001849 | -0.000288 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002032 | -0.000020 |
| employment_type | salaried | 33473 | 4432 | -0.002516 | +0.000011 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.002695 | -0.001274 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.