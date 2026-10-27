# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 267 | 0.3857677902621723 | 0.36329588014981273 |
| 0.200000 | 21 | 414 | 0.2584541062801932 | 0.20772946859903382 |
| 0.323686 | -4 | 525 | 0.1219047619047619 | 0.1295238095238095 |
| 0.400000 | 3 | 551 | 0.10163339382940109 | 0.09618874773139746 |
| 0.500000 | -1 | 562 | 0.060498220640569395 | 0.06227758007117438 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29938 | 544 | +0.008354 | +0.001291 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.006113 | -0.001313 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003981 | -0.002656 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.002815 | +0.001375 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.002427 | +0.000192 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.001416 | -0.001241 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000040 | -0.000247 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000141 | -0.000297 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000486 | -0.000096 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000837 | +0.000049 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001156 | +0.000120 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001171 | +0.000110 |
| employment_type | variable_income | 15540 | 2180 | -0.001207 | -0.000421 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001540 | +0.000005 |
| aux_account_months | zero | 21000 | 2461 | -0.001600 | +0.000041 |
| employment_type | salaried | 33473 | 4432 | -0.001878 | +0.000110 |
| employment_type | self_employed | 20987 | 2895 | -0.002680 | +0.000165 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002721 | +0.000083 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.