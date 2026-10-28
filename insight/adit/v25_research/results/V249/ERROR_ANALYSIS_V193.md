# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 13 | 270 | 0.3962962962962963 | 0.34814814814814815 |
| 0.200000 | -3 | 410 | 0.23170731707317074 | 0.23902439024390243 |
| 0.323686 | 0 | 478 | 0.11715481171548117 | 0.11715481171548117 |
| 0.400000 | 5 | 522 | 0.09961685823754789 | 0.09003831417624521 |
| 0.500000 | 5 | 504 | 0.07738095238095238 | 0.06746031746031746 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003565 | -0.000583 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002207 | -0.000175 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001654 | -0.000198 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000528 | +0.000082 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000322 | -0.000130 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000040 | -0.000393 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000650 | -0.000218 |
| aux_account_months | zero | 21000 | 2461 | -0.000736 | -0.000315 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.000943 | -0.001535 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001709 | -0.000180 |
| employment_type | variable_income | 15540 | 2180 | -0.001792 | -0.000364 |
| employment_type | self_employed | 20987 | 2895 | -0.002419 | -0.000136 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002451 | -0.000067 |
| employment_type | salaried | 33473 | 4432 | -0.002590 | -0.000156 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002791 | -0.000190 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002923 | -0.000143 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.003001 | -0.000136 |
| baseline_risk_band | under005 | 29938 | 544 | -0.004043 | -0.001040 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.