# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 254 | 0.3937007874015748 | 0.36220472440944884 |
| 0.200000 | 1 | 401 | 0.23192019950124687 | 0.22942643391521197 |
| 0.323686 | 3 | 478 | 0.12552301255230125 | 0.1192468619246862 |
| 0.400000 | -5 | 511 | 0.0821917808219178 | 0.09197651663405088 |
| 0.500000 | 2 | 496 | 0.07459677419354839 | 0.07056451612903226 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003154 | -0.000232 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003023 | -0.000188 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.002028 | +0.000495 |
| aux_account_months | zero | 21000 | 2461 | +0.000788 | -0.000378 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000016 | -0.000165 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000590 | -0.000092 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000901 | -0.000171 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001539 | -0.000197 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001564 | -0.000030 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.001663 | -0.001768 |
| employment_type | self_employed | 20987 | 2895 | -0.001728 | -0.000141 |
| employment_type | variable_income | 15540 | 2180 | -0.001936 | -0.000385 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002150 | -0.000088 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002153 | -0.000102 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002574 | -0.000148 |
| employment_type | salaried | 33473 | 4432 | -0.003235 | -0.000101 |
| baseline_risk_band | under005 | 29965 | 546 | -0.004561 | -0.000842 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005896 | -0.000574 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.