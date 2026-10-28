# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 250 | 0.372 | 0.332 |
| 0.200000 | -6 | 381 | 0.2178477690288714 | 0.2335958005249344 |
| 0.323686 | 5 | 437 | 0.13043478260869565 | 0.11899313501144165 |
| 0.400000 | 1 | 474 | 0.09071729957805907 | 0.08860759493670886 |
| 0.500000 | 5 | 448 | 0.07589285714285714 | 0.06473214285714286 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003509 | -0.000108 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003180 | -0.000135 |
| aux_account_months | zero | 21000 | 2461 | +0.001511 | -0.000301 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000980 | -0.000175 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000447 | -0.000267 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000374 | -0.000020 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000110 | -0.000094 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000696 | +0.000331 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001195 | -0.001617 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001203 | -0.000106 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001245 | -0.000066 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001394 | -0.000034 |
| employment_type | self_employed | 20987 | 2895 | -0.001437 | +0.000046 |
| employment_type | variable_income | 15540 | 2180 | -0.001635 | -0.000417 |
| employment_type | salaried | 33473 | 4432 | -0.002447 | -0.000070 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.002447 | +0.000051 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002500 | +0.000035 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.006314 | -0.000228 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.