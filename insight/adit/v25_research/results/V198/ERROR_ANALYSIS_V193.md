# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 209 | 0.3827751196172249 | 0.3588516746411483 |
| 0.200000 | 19 | 337 | 0.258160237388724 | 0.20178041543026706 |
| 0.323686 | 0 | 390 | 0.1282051282051282 | 0.1282051282051282 |
| 0.400000 | 14 | 412 | 0.10679611650485436 | 0.07281553398058252 |
| 0.500000 | -9 | 425 | 0.058823529411764705 | 0.08 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29938 | 544 | +0.010139 | +0.002741 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.005536 | -0.000488 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.004960 | -0.000241 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.003463 | +0.001389 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001485 | -0.000050 |
| aux_account_months | zero | 21000 | 2461 | +0.001349 | +0.000195 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000914 | +0.000205 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000611 | +0.000163 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000244 | +0.000215 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.001204 | +0.000428 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001219 | +0.000246 |
| employment_type | variable_income | 15540 | 2180 | -0.001320 | -0.000228 |
| employment_type | salaried | 33473 | 4432 | -0.001699 | +0.000347 |
| employment_type | self_employed | 20987 | 2895 | -0.001840 | +0.000114 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002048 | +0.000085 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002481 | +0.000175 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002530 | +0.000148 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004131 | +0.000506 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.