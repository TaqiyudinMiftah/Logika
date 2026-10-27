# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 330 | 0.36666666666666664 | 0.36666666666666664 |
| 0.200000 | 28 | 516 | 0.2364341085271318 | 0.1821705426356589 |
| 0.323686 | -22 | 636 | 0.10849056603773585 | 0.1430817610062893 |
| 0.400000 | -9 | 741 | 0.0931174089068826 | 0.10526315789473684 |
| 0.500000 | -21 | 776 | 0.055412371134020616 | 0.08247422680412371 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.004912 | -0.000066 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.002871 | +0.000083 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.002500 | -0.000761 |
| employment_type | salaried | 33473 | 4432 | -0.000258 | -0.000631 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000644 | -0.000885 |
| orders90_exposure_band | zero | 3428 | 106 | -0.000926 | +0.000446 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001925 | -0.000717 |
| employment_type | variable_income | 15540 | 2180 | -0.002493 | -0.000606 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002499 | -0.000693 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002627 | -0.000856 |
| employment_type | self_employed | 20987 | 2895 | -0.002997 | -0.000713 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.003223 | -0.000333 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003723 | -0.000431 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003962 | -0.004381 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.004010 | -0.000823 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.004176 | -0.000219 |
| aux_account_months | zero | 21000 | 2461 | -0.004255 | -0.000105 |
| aux_allocation_count | zero | 23732 | 2895 | -0.004490 | -0.000177 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.006746 | -0.000985 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.007123 | -0.006665 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.009643 | -0.002456 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.009929 | -0.001416 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.