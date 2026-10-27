# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 269 | 0.3828996282527881 | 0.35687732342007433 |
| 0.200000 | 0 | 424 | 0.22169811320754718 | 0.22169811320754718 |
| 0.323686 | -9 | 503 | 0.11332007952286283 | 0.1312127236580517 |
| 0.400000 | 2 | 550 | 0.08545454545454545 | 0.08181818181818182 |
| 0.500000 | -3 | 515 | 0.06601941747572816 | 0.07184466019417475 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29938 | 544 | +0.007327 | -0.000266 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002737 | -0.000026 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.002371 | +0.000065 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002325 | -0.000101 |
| aux_account_months | zero | 21000 | 2461 | +0.000893 | -0.000276 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000700 | -0.000082 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000228 | +0.000001 |
| employment_type | variable_income | 15540 | 2180 | -0.000196 | -0.000532 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.000860 | -0.001624 |
| employment_type | self_employed | 20987 | 2895 | -0.001049 | +0.000055 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001380 | +0.000623 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001506 | +0.000175 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.002282 | +0.001586 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002461 | +0.000041 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002512 | +0.000046 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003071 | -0.000899 |
| employment_type | salaried | 33473 | 4432 | -0.003417 | +0.000117 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.003740 | -0.000062 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.