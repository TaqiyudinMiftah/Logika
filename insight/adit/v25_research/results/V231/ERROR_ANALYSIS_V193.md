# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 11 | 243 | 0.3868312757201646 | 0.34156378600823045 |
| 0.200000 | -6 | 370 | 0.22162162162162163 | 0.23783783783783785 |
| 0.323686 | 9 | 416 | 0.13701923076923078 | 0.11538461538461539 |
| 0.400000 | 8 | 461 | 0.09327548806941431 | 0.07592190889370933 |
| 0.500000 | 2 | 432 | 0.07407407407407407 | 0.06944444444444445 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003372 | +0.000002 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003123 | -0.000021 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001771 | -0.000136 |
| aux_account_months | zero | 21000 | 2461 | +0.001482 | -0.000179 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.001127 | +0.000731 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000564 | +0.000045 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000048 | -0.000023 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000403 | -0.000068 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000603 | -0.000028 |
| employment_type | self_employed | 20987 | 2895 | -0.000610 | +0.000052 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000690 | +0.000325 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000797 | -0.000002 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000898 | +0.000008 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001476 | -0.001186 |
| employment_type | variable_income | 15540 | 2180 | -0.001545 | -0.000306 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002218 | -0.000455 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002705 | +0.000022 |
| employment_type | salaried | 33473 | 4432 | -0.002997 | -0.000011 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.