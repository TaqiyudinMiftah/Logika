# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -6 | 302 | 0.36423841059602646 | 0.3841059602649007 |
| 0.200000 | -1 | 480 | 0.24166666666666667 | 0.24375 |
| 0.323686 | 6 | 599 | 0.14023372287145242 | 0.1302170283806344 |
| 0.400000 | -2 | 668 | 0.09730538922155689 | 0.10029940119760479 |
| 0.500000 | -17 | 702 | 0.0641025641025641 | 0.08831908831908832 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.004617 | -0.000992 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000985 | -0.002335 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000389 | -0.000605 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000170 | -0.000414 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.001047 | -0.007292 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001128 | -0.000956 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001523 | -0.000927 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001809 | -0.000935 |
| employment_type | salaried | 33473 | 4432 | -0.001918 | -0.001007 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001943 | -0.001020 |
| employment_type | variable_income | 15540 | 2180 | -0.001944 | -0.000563 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.002988 | -0.003165 |
| aux_account_months | zero | 21000 | 2461 | -0.003485 | -0.000437 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003657 | -0.000508 |
| employment_type | self_employed | 20987 | 2895 | -0.004020 | -0.000874 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004928 | -0.001280 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.007431 | -0.003347 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.010412 | -0.003382 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.