# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 245 | 0.3673469387755102 | 0.33877551020408164 |
| 0.200000 | -5 | 364 | 0.22527472527472528 | 0.23901098901098902 |
| 0.323686 | 12 | 425 | 0.13647058823529412 | 0.10823529411764705 |
| 0.400000 | 6 | 465 | 0.09462365591397849 | 0.08172043010752689 |
| 0.500000 | 2 | 441 | 0.07482993197278912 | 0.07029478458049887 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003152 | -0.000064 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002892 | -0.000107 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002753 | -0.000167 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001546 | +0.000034 |
| aux_account_months | zero | 21000 | 2461 | +0.001403 | -0.000279 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000396 | +0.000394 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000264 | -0.000058 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.000134 | +0.000228 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000558 | -0.000032 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000698 | +0.000000 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000906 | -0.000192 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000936 | -0.000080 |
| employment_type | self_employed | 20987 | 2895 | -0.001232 | +0.000047 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001443 | -0.001309 |
| employment_type | variable_income | 15540 | 2180 | -0.001520 | -0.000340 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002086 | +0.000041 |
| employment_type | salaried | 33473 | 4432 | -0.002569 | -0.000043 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003962 | -0.000293 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.