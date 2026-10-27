# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 13 | 245 | 0.3877551020408163 | 0.3346938775510204 |
| 0.200000 | -3 | 365 | 0.2273972602739726 | 0.2356164383561644 |
| 0.323686 | 9 | 424 | 0.1320754716981132 | 0.11084905660377359 |
| 0.400000 | 5 | 453 | 0.09271523178807947 | 0.08167770419426049 |
| 0.500000 | 4 | 428 | 0.07476635514018691 | 0.06542056074766354 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002782 | +0.000015 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002659 | -0.000013 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002140 | -0.000106 |
| aux_account_months | zero | 21000 | 2461 | +0.000928 | -0.000167 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000533 | +0.000822 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000441 | +0.000045 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000029 | +0.000398 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000086 | -0.000065 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000261 | -0.000011 |
| employment_type | self_employed | 20987 | 2895 | -0.000789 | +0.000070 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000849 | -0.000009 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000959 | +0.000017 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000967 | -0.000057 |
| employment_type | variable_income | 15540 | 2180 | -0.001030 | -0.000302 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002237 | +0.000060 |
| employment_type | salaried | 33473 | 4432 | -0.002262 | +0.000014 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.002346 | -0.000976 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003773 | -0.000447 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.