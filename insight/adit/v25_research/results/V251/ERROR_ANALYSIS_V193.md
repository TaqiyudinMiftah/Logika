# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 13 | 243 | 0.39094650205761317 | 0.3374485596707819 |
| 0.200000 | 1 | 374 | 0.22459893048128343 | 0.22192513368983957 |
| 0.323686 | 4 | 441 | 0.12471655328798185 | 0.11564625850340136 |
| 0.400000 | 7 | 477 | 0.09014675052410902 | 0.07547169811320754 |
| 0.500000 | 0 | 452 | 0.06858407079646017 | 0.06858407079646017 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002798 | -0.000106 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002337 | -0.000140 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.001406 | +0.000611 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001331 | -0.000044 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000963 | -0.000214 |
| aux_account_months | zero | 21000 | 2461 | +0.000918 | -0.000314 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000070 | -0.001446 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000393 | -0.000094 |
| employment_type | variable_income | 15540 | 2180 | -0.000491 | -0.000369 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001193 | -0.000030 |
| baseline_risk_band | under005 | 29938 | 544 | -0.001294 | -0.000544 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001326 | -0.000051 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001375 | -0.000076 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001716 | -0.000123 |
| employment_type | self_employed | 20987 | 2895 | -0.002074 | -0.000038 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002812 | +0.000012 |
| employment_type | salaried | 33473 | 4432 | -0.002850 | -0.000078 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003299 | -0.000543 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.