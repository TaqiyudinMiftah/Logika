# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 25 | 324 | 0.4104938271604938 | 0.3333333333333333 |
| 0.200000 | 16 | 520 | 0.2423076923076923 | 0.21153846153846154 |
| 0.323686 | -4 | 647 | 0.12364760432766615 | 0.12982998454404945 |
| 0.400000 | 4 | 679 | 0.09572901325478646 | 0.0898379970544919 |
| 0.500000 | -14 | 685 | 0.06423357664233577 | 0.08467153284671533 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.004626 | +0.000086 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.004189 | +0.000349 |
| baseline_risk_band | under005 | 29938 | 544 | +0.004164 | +0.002639 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003451 | +0.000337 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002524 | +0.000310 |
| aux_account_months | zero | 21000 | 2461 | +0.001658 | +0.000148 |
| employment_type | variable_income | 15540 | 2180 | -0.000102 | -0.000253 |
| employment_type | self_employed | 20987 | 2895 | -0.000330 | +0.000258 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000359 | +0.001697 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000402 | +0.000105 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001483 | +0.000293 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001881 | +0.000278 |
| employment_type | salaried | 33473 | 4432 | -0.001882 | +0.000408 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002319 | +0.000065 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002727 | +0.000242 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002886 | +0.000752 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.003451 | +0.002876 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.006238 | -0.001868 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.