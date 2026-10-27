# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 245 | 0.3673469387755102 | 0.34285714285714286 |
| 0.200000 | -2 | 372 | 0.22580645161290322 | 0.23118279569892472 |
| 0.323686 | 13 | 429 | 0.14685314685314685 | 0.11655011655011654 |
| 0.400000 | 7 | 454 | 0.09030837004405286 | 0.07488986784140969 |
| 0.500000 | 0 | 447 | 0.07158836689038031 | 0.07158836689038031 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003527 | -0.000065 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003480 | -0.000095 |
| aux_account_months | zero | 21000 | 2461 | +0.002022 | -0.000270 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001526 | -0.000203 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001343 | +0.000037 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000272 | -0.000069 |
| employment_type | self_employed | 20987 | 2895 | -0.000057 | +0.000006 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000329 | -0.000100 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000440 | -0.000028 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000455 | -0.000065 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000487 | +0.000378 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000518 | +0.000142 |
| employment_type | variable_income | 15540 | 2180 | -0.000996 | -0.000329 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001823 | -0.000016 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001848 | -0.000027 |
| employment_type | salaried | 33473 | 4432 | -0.002775 | -0.000071 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.004469 | -0.001614 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005043 | -0.000268 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.