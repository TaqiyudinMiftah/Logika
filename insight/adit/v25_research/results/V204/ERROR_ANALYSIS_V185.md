# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -7 | 193 | 0.3471502590673575 | 0.38341968911917096 |
| 0.200000 | 14 | 293 | 0.2354948805460751 | 0.18771331058020477 |
| 0.323686 | 12 | 378 | 0.16137566137566137 | 0.12962962962962962 |
| 0.400000 | 12 | 396 | 0.1111111111111111 | 0.08080808080808081 |
| 0.500000 | -13 | 375 | 0.05333333333333334 | 0.088 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6079 | 3488 | +0.003727 | -0.002349 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001804 | +0.000106 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001692 | -0.000286 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001572 | +0.000307 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001437 | -0.000198 |
| aux_account_months | zero | 21000 | 2461 | +0.001367 | -0.000326 |
| employment_type | variable_income | 15540 | 2180 | +0.000739 | -0.000267 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.000330 | -0.000780 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000320 | +0.000073 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000291 | +0.001194 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000215 | +0.000049 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000164 | +0.000014 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000466 | +0.000538 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000762 | -0.000269 |
| employment_type | salaried | 33473 | 4432 | -0.000973 | +0.000129 |
| employment_type | self_employed | 20987 | 2895 | -0.001300 | -0.000209 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001321 | -0.000127 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001725 | -0.000340 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.