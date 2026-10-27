# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 247 | 0.3765182186234818 | 0.3481781376518219 |
| 0.200000 | -2 | 382 | 0.22774869109947643 | 0.23298429319371727 |
| 0.323686 | 5 | 437 | 0.14187643020594964 | 0.13043478260869565 |
| 0.400000 | 9 | 474 | 0.08438818565400844 | 0.06540084388185655 |
| 0.500000 | 0 | 456 | 0.06578947368421052 | 0.06578947368421052 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003362 | -0.000044 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002768 | -0.000052 |
| aux_account_months | zero | 21000 | 2461 | +0.001666 | -0.000215 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000520 | +0.000125 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000260 | +0.000005 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000210 | -0.000263 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000097 | -0.000076 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000420 | -0.000025 |
| employment_type | variable_income | 15540 | 2180 | -0.000743 | -0.000355 |
| employment_type | self_employed | 20987 | 2895 | -0.000872 | -0.000040 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000945 | -0.000084 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000947 | -0.000114 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001327 | -0.000133 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001403 | -0.000108 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001555 | -0.000344 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001674 | -0.001882 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.001728 | +0.000287 |
| employment_type | salaried | 33473 | 4432 | -0.002627 | -0.000096 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.