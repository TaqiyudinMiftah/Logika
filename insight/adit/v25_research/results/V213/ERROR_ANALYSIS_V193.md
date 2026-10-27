# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 26 | 0.5 | 0.4230769230769231 |
| 0.200000 | 1 | 54 | 0.2962962962962963 | 0.2777777777777778 |
| 0.323686 | -2 | 70 | 0.12857142857142856 | 0.15714285714285714 |
| 0.400000 | 6 | 55 | 0.10909090909090909 | 0.0 |
| 0.500000 | 0 | 71 | 0.07042253521126761 | 0.07042253521126761 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.006135 | +0.000466 |
| baseline_risk_band | under005 | 29938 | 544 | +0.002969 | +0.000545 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.000645 | +0.000555 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000463 | +0.000164 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000432 | +0.000106 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000148 | +0.000095 |
| employment_type | self_employed | 20987 | 2895 | +0.000083 | +0.000094 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000092 | +0.000095 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000097 | +0.000098 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000158 | +0.000098 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000208 | -0.000143 |
| aux_account_months | zero | 21000 | 2461 | -0.000270 | +0.000083 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000370 | +0.000099 |
| employment_type | variable_income | 15540 | 2180 | -0.000415 | +0.000031 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000499 | +0.000096 |
| employment_type | salaried | 33473 | 4432 | -0.000666 | +0.000109 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000888 | +0.000657 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001095 | +0.000078 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.