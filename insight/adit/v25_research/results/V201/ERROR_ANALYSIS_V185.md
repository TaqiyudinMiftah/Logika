# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 248 | 0.3709677419354839 | 0.34274193548387094 |
| 0.200000 | -4 | 370 | 0.21621621621621623 | 0.22702702702702704 |
| 0.323686 | 18 | 434 | 0.14055299539170507 | 0.09907834101382489 |
| 0.400000 | 5 | 454 | 0.09030837004405286 | 0.07929515418502203 |
| 0.500000 | 0 | 445 | 0.07415730337078652 | 0.07415730337078652 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003461 | -0.000128 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.003324 | -0.000137 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003296 | -0.000072 |
| aux_account_months | zero | 21000 | 2461 | +0.001439 | -0.000306 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000869 | +0.000362 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000783 | +0.000055 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000369 | -0.000050 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000239 | +0.000068 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000125 | -0.000015 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000035 | +0.000023 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000061 | -0.000074 |
| employment_type | variable_income | 15540 | 2180 | -0.000153 | -0.000302 |
| employment_type | self_employed | 20987 | 2895 | -0.000154 | +0.000084 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.001190 | -0.001344 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.001865 | +0.000398 |
| employment_type | salaried | 33473 | 4432 | -0.002598 | -0.000053 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002899 | +0.000110 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004341 | -0.000239 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.