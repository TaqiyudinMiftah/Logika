# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 245 | 0.37551020408163266 | 0.3510204081632653 |
| 0.200000 | -2 | 385 | 0.22337662337662337 | 0.22857142857142856 |
| 0.323686 | 7 | 440 | 0.12727272727272726 | 0.11136363636363636 |
| 0.400000 | 7 | 481 | 0.09563409563409564 | 0.08108108108108109 |
| 0.500000 | 3 | 458 | 0.07641921397379912 | 0.06986899563318777 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002866 | -0.000180 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002731 | -0.000206 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.002388 | -0.001348 |
| aux_account_months | zero | 21000 | 2461 | +0.001051 | -0.000332 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001032 | -0.000042 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000579 | -0.000632 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000548 | -0.000136 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000065 | -0.000163 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000575 | -0.000137 |
| employment_type | variable_income | 15540 | 2180 | -0.000641 | -0.000383 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001542 | -0.000067 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001544 | -0.000105 |
| employment_type | self_employed | 20987 | 2895 | -0.001660 | -0.000015 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001661 | -0.000076 |
| employment_type | salaried | 33473 | 4432 | -0.001843 | -0.000115 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.001941 | +0.000115 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003425 | -0.000093 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003754 | -0.000683 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.