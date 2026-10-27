# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 146 | 0.3972602739726027 | 0.3356164383561644 |
| 0.200000 | 10 | 241 | 0.23651452282157676 | 0.1950207468879668 |
| 0.323686 | -13 | 292 | 0.10273972602739725 | 0.14726027397260275 |
| 0.400000 | -6 | 331 | 0.08459214501510574 | 0.1027190332326284 |
| 0.500000 | -3 | 330 | 0.05454545454545454 | 0.06363636363636363 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.011013 | +0.002048 |
| baseline_risk_band | under005 | 29853 | 535 | +0.008520 | +0.001973 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.002260 | +0.001791 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001782 | -0.000398 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001366 | +0.000588 |
| employment_type | variable_income | 15540 | 2180 | +0.000659 | +0.000342 |
| employment_type | self_employed | 20987 | 2895 | +0.000211 | +0.000215 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000331 | +0.000215 |
| employment_type | salaried | 33473 | 4432 | -0.000435 | +0.000382 |
| aux_account_months | zero | 21000 | 2461 | -0.000518 | +0.000359 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000594 | +0.000430 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000629 | +0.000065 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001082 | +0.000342 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001160 | +0.000536 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001204 | +0.000378 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001613 | +0.000312 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001990 | +0.000295 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.003071 | +0.001010 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.