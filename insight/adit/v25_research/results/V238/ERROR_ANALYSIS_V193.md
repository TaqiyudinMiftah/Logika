# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 11 | 289 | 0.39100346020761245 | 0.35294117647058826 |
| 0.200000 | 15 | 490 | 0.24285714285714285 | 0.21224489795918366 |
| 0.323686 | -7 | 585 | 0.11452991452991453 | 0.1264957264957265 |
| 0.400000 | -1 | 597 | 0.08542713567839195 | 0.08710217755443886 |
| 0.500000 | -4 | 609 | 0.06896551724137931 | 0.0755336617405583 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.004955 | -0.000887 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.004165 | +0.001325 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.003438 | +0.000397 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002648 | +0.000072 |
| baseline_risk_band | under005 | 29938 | 544 | +0.001694 | +0.001177 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001645 | -0.000023 |
| aux_account_months | zero | 21000 | 2461 | +0.000803 | -0.000229 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000583 | +0.002626 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000553 | +0.000027 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000118 | +0.000056 |
| employment_type | variable_income | 15540 | 2180 | -0.000311 | -0.000368 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000683 | -0.001365 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001419 | +0.000370 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002428 | +0.000090 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002458 | +0.000227 |
| employment_type | salaried | 33473 | 4432 | -0.002568 | +0.000246 |
| employment_type | self_employed | 20987 | 2895 | -0.002761 | +0.000185 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002955 | +0.000207 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.