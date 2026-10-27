# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 17 | 0.47058823529411764 | 0.4117647058823529 |
| 0.200000 | 0 | 27 | 0.2222222222222222 | 0.2222222222222222 |
| 0.323686 | -3 | 48 | 0.10416666666666667 | 0.16666666666666666 |
| 0.400000 | 4 | 34 | 0.14705882352941177 | 0.029411764705882353 |
| 0.500000 | -2 | 41 | 0.024390243902439025 | 0.07317073170731707 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.002647 | +0.000157 |
| baseline_risk_band | under005 | 29938 | 544 | +0.001339 | +0.000454 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000597 | +0.000016 |
| aux_account_months | zero | 21000 | 2461 | +0.000389 | +0.000043 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000308 | +0.000045 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000236 | +0.000035 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000211 | +0.000193 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.000184 | +0.000261 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000144 | +0.000040 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000077 | +0.000042 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000013 | +0.000050 |
| employment_type | self_employed | 20987 | 2895 | -0.000014 | +0.000036 |
| employment_type | salaried | 33473 | 4432 | -0.000034 | +0.000058 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000088 | +0.000045 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000191 | +0.000050 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000266 | -0.000055 |
| employment_type | variable_income | 15540 | 2180 | -0.000439 | +0.000000 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000785 | +0.000088 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.