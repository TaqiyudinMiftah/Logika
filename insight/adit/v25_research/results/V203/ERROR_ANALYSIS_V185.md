# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 76 | 0.42105263157894735 | 0.35526315789473684 |
| 0.200000 | -6 | 132 | 0.22727272727272727 | 0.2727272727272727 |
| 0.323686 | 5 | 160 | 0.14375 | 0.1125 |
| 0.400000 | 8 | 160 | 0.1 | 0.05 |
| 0.500000 | 1 | 161 | 0.06832298136645963 | 0.062111801242236024 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29965 | 546 | +0.002450 | +0.001544 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002301 | +0.000064 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001585 | +0.000082 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001435 | +0.000057 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001215 | +0.000065 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.001122 | +0.000062 |
| aux_account_months | zero | 21000 | 2461 | +0.001119 | +0.000038 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000916 | -0.000792 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000686 | -0.000052 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000266 | +0.000080 |
| employment_type | variable_income | 15540 | 2180 | -0.000001 | +0.000072 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000393 | +0.000104 |
| employment_type | salaried | 33473 | 4432 | -0.000486 | +0.000046 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000587 | +0.000046 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000680 | +0.000029 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.000762 | +0.000161 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000764 | +0.000033 |
| employment_type | self_employed | 20987 | 2895 | -0.001548 | +0.000028 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.