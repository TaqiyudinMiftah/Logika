# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 307 | 0.3745928338762215 | 0.3550488599348534 |
| 0.200000 | 7 | 475 | 0.2231578947368421 | 0.20842105263157895 |
| 0.323686 | -10 | 586 | 0.11774744027303755 | 0.1348122866894198 |
| 0.400000 | -3 | 601 | 0.08485856905158069 | 0.08985024958402663 |
| 0.500000 | -6 | 583 | 0.058319039451114926 | 0.0686106346483705 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.004358 | -0.000959 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003787 | +0.000120 |
| baseline_risk_band | under005 | 29938 | 544 | +0.003129 | +0.001569 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003057 | -0.000090 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.002170 | +0.000362 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001772 | +0.000087 |
| aux_account_months | zero | 21000 | 2461 | +0.001087 | -0.000264 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000048 | +0.001509 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000401 | +0.000096 |
| employment_type | variable_income | 15540 | 2180 | -0.000953 | -0.000442 |
| employment_type | self_employed | 20987 | 2895 | -0.001215 | +0.000010 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001832 | -0.000124 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001846 | -0.000071 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001972 | -0.000031 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002185 | +0.000052 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002872 | +0.000086 |
| employment_type | salaried | 33473 | 4432 | -0.003262 | +0.000182 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004146 | +0.000025 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.