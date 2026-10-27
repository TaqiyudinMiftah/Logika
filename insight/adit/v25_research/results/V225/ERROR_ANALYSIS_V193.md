# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 12 | 244 | 0.38524590163934425 | 0.3360655737704918 |
| 0.200000 | 0 | 362 | 0.23204419889502761 | 0.23204419889502761 |
| 0.323686 | 7 | 422 | 0.13270142180094788 | 0.11611374407582939 |
| 0.400000 | 3 | 457 | 0.08971553610503283 | 0.08315098468271334 |
| 0.500000 | 3 | 436 | 0.07339449541284404 | 0.06651376146788991 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003005 | +0.000027 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002944 | +0.000048 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002265 | -0.000079 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.001820 | +0.001025 |
| aux_account_months | zero | 21000 | 2461 | +0.001076 | -0.000118 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000306 | -0.000001 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000278 | +0.000055 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000132 | +0.000008 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000486 | -0.000049 |
| employment_type | variable_income | 15540 | 2180 | -0.000510 | -0.000289 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000764 | -0.000001 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000866 | +0.000461 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001131 | +0.000023 |
| employment_type | self_employed | 20987 | 2895 | -0.001300 | +0.000082 |
| employment_type | salaried | 33473 | 4432 | -0.002114 | +0.000037 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.002699 | -0.000840 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002705 | +0.000073 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004664 | -0.000484 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.