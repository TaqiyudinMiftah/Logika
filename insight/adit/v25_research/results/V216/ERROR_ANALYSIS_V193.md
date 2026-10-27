# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 252 | 0.373015873015873 | 0.34523809523809523 |
| 0.200000 | 12 | 392 | 0.2423469387755102 | 0.21173469387755103 |
| 0.323686 | 8 | 480 | 0.12708333333333333 | 0.11041666666666666 |
| 0.400000 | 2 | 521 | 0.09213051823416507 | 0.08829174664107485 |
| 0.500000 | -4 | 493 | 0.07099391480730223 | 0.07910750507099391 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003611 | -0.000042 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003371 | -0.000073 |
| baseline_risk_band | under005 | 29938 | 544 | +0.002706 | +0.000685 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.002129 | -0.000183 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001359 | -0.000266 |
| aux_account_months | zero | 21000 | 2461 | +0.001083 | -0.000230 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000923 | +0.000072 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000401 | +0.000109 |
| employment_type | self_employed | 20987 | 2895 | -0.000126 | +0.000010 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000406 | +0.000105 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000853 | +0.000029 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.001207 | +0.000465 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001208 | -0.000049 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001258 | +0.000062 |
| employment_type | variable_income | 15540 | 2180 | -0.001781 | -0.000375 |
| employment_type | salaried | 33473 | 4432 | -0.002870 | +0.000120 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003090 | -0.000184 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004785 | +0.000313 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.