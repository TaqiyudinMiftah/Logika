# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -8 | 266 | 0.3533834586466165 | 0.38345864661654133 |
| 0.200000 | 9 | 398 | 0.21105527638190955 | 0.1884422110552764 |
| 0.323686 | -38 | 520 | 0.09423076923076923 | 0.1673076923076923 |
| 0.400000 | 8 | 605 | 0.11239669421487604 | 0.09917355371900827 |
| 0.500000 | -14 | 625 | 0.0624 | 0.0848 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003488 | +0.000048 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001237 | -0.000764 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000236 | -0.002973 |
| employment_type | salaried | 33473 | 4432 | +0.000149 | -0.000597 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001547 | -0.003658 |
| employment_type | self_employed | 20987 | 2895 | -0.002251 | -0.000900 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002380 | -0.000194 |
| employment_type | variable_income | 15540 | 2180 | -0.002671 | -0.000178 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002904 | -0.000485 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003268 | -0.000169 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.003425 | -0.000576 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003633 | -0.000149 |
| aux_account_months | zero | 21000 | 2461 | -0.003798 | -0.000233 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.004157 | -0.000562 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.004508 | -0.000545 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004846 | -0.000574 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.005739 | -0.002491 |
| baseline_risk_band | under005 | 29853 | 535 | -0.010935 | -0.004999 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.