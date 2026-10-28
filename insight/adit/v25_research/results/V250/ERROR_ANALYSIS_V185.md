# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 269 | 0.379182156133829 | 0.34944237918215615 |
| 0.200000 | -4 | 404 | 0.2202970297029703 | 0.2301980198019802 |
| 0.323686 | 7 | 479 | 0.1315240083507307 | 0.11691022964509394 |
| 0.400000 | 3 | 511 | 0.09393346379647749 | 0.08806262230919765 |
| 0.500000 | 3 | 514 | 0.07392996108949416 | 0.06809338521400778 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003912 | -0.000261 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003734 | -0.000266 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.002348 | -0.000044 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001824 | -0.000168 |
| aux_account_months | zero | 21000 | 2461 | +0.001108 | -0.000391 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000441 | -0.000241 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.000598 | -0.001582 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000950 | -0.001011 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001293 | -0.000254 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001539 | -0.000731 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001719 | -0.000174 |
| employment_type | variable_income | 15540 | 2180 | -0.001748 | -0.000411 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001823 | -0.000218 |
| employment_type | self_employed | 20987 | 2895 | -0.002046 | -0.000184 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002119 | -0.000184 |
| employment_type | salaried | 33473 | 4432 | -0.002400 | -0.000187 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003129 | -0.000115 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004152 | -0.000826 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.