# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 295 | 0.3593220338983051 | 0.3423728813559322 |
| 0.200000 | 6 | 471 | 0.2356687898089172 | 0.2229299363057325 |
| 0.323686 | 5 | 561 | 0.13368983957219252 | 0.12477718360071301 |
| 0.400000 | -3 | 586 | 0.08191126279863481 | 0.08703071672354949 |
| 0.500000 | -2 | 583 | 0.07375643224699828 | 0.07718696397941681 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29965 | 546 | +0.005669 | +0.002255 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.004015 | +0.001151 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.003492 | +0.000338 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003411 | +0.000389 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002999 | +0.000208 |
| aux_account_months | zero | 21000 | 2461 | +0.001939 | -0.000044 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.001678 | +0.001736 |
| employment_type | variable_income | 15540 | 2180 | +0.000048 | +0.000036 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000401 | +0.000316 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000622 | +0.000150 |
| employment_type | self_employed | 20987 | 2895 | -0.000979 | +0.000245 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001318 | +0.000145 |
| employment_type | salaried | 33473 | 4432 | -0.001644 | +0.000341 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001739 | +0.000294 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002131 | +0.000345 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.002476 | +0.001560 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002505 | +0.000453 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.007925 | +0.000042 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.