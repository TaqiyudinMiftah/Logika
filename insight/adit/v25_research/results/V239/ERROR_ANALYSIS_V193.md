# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 294 | 0.36394557823129253 | 0.3435374149659864 |
| 0.200000 | 6 | 480 | 0.23958333333333334 | 0.22708333333333333 |
| 0.323686 | 0 | 566 | 0.13074204946996468 | 0.13074204946996468 |
| 0.400000 | 0 | 592 | 0.08614864864864864 | 0.08614864864864864 |
| 0.500000 | -2 | 586 | 0.07337883959044368 | 0.07679180887372014 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29938 | 544 | +0.006527 | +0.002272 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.005335 | +0.001185 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.003716 | +0.000336 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003180 | +0.000406 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002488 | +0.000230 |
| aux_account_months | zero | 21000 | 2461 | +0.001497 | -0.000013 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.001328 | +0.001690 |
| employment_type | variable_income | 15540 | 2180 | -0.000261 | +0.000030 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000399 | +0.000135 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000732 | +0.000325 |
| employment_type | self_employed | 20987 | 2895 | -0.001339 | +0.000244 |
| employment_type | salaried | 33473 | 4432 | -0.001771 | +0.000345 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001860 | +0.000145 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002231 | +0.000293 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002559 | +0.000435 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002654 | +0.000337 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.002984 | +0.001543 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005764 | -0.000057 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.