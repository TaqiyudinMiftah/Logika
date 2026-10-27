# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 241 | 0.3775933609958506 | 0.34024896265560167 |
| 0.200000 | -8 | 368 | 0.22010869565217392 | 0.2418478260869565 |
| 0.323686 | 8 | 417 | 0.13189448441247004 | 0.11270983213429256 |
| 0.400000 | 11 | 455 | 0.0989010989010989 | 0.07472527472527472 |
| 0.500000 | 3 | 431 | 0.07424593967517401 | 0.06728538283062645 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003543 | +0.000010 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003238 | -0.000006 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001925 | -0.000172 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.001896 | +0.000653 |
| aux_account_months | zero | 21000 | 2461 | +0.001867 | -0.000171 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001339 | +0.000038 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000974 | +0.000088 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000062 | -0.000003 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000735 | -0.000070 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000739 | -0.000035 |
| employment_type | self_employed | 20987 | 2895 | -0.000817 | +0.000037 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000890 | -0.000003 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001193 | +0.000374 |
| employment_type | variable_income | 15540 | 2180 | -0.001549 | -0.000295 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002086 | -0.000006 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.002477 | -0.001273 |
| employment_type | salaried | 33473 | 4432 | -0.002576 | -0.000007 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002882 | -0.000369 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.