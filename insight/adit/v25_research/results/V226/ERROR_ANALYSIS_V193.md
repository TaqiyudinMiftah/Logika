# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 252 | 0.36904761904761907 | 0.3492063492063492 |
| 0.200000 | -5 | 380 | 0.22105263157894736 | 0.23421052631578948 |
| 0.323686 | 5 | 424 | 0.14150943396226415 | 0.12971698113207547 |
| 0.400000 | 6 | 472 | 0.0847457627118644 | 0.07203389830508475 |
| 0.500000 | 0 | 447 | 0.07158836689038031 | 0.07158836689038031 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002692 | -0.000038 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002458 | -0.000061 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001127 | -0.000152 |
| aux_account_months | zero | 21000 | 2461 | +0.000738 | -0.000205 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000618 | -0.000083 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000404 | +0.000033 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000507 | +0.000613 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000595 | -0.000083 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000869 | +0.000051 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000954 | -0.000059 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001050 | -0.000102 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001122 | -0.000041 |
| employment_type | variable_income | 15540 | 2180 | -0.001407 | -0.000336 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001555 | -0.000551 |
| employment_type | self_employed | 20987 | 2895 | -0.001563 | +0.000027 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.002077 | -0.001399 |
| employment_type | salaried | 33473 | 4432 | -0.002638 | -0.000058 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002978 | +0.000004 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.