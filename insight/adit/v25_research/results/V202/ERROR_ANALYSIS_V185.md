# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 246 | 0.3699186991869919 | 0.34146341463414637 |
| 0.200000 | -5 | 359 | 0.2200557103064067 | 0.233983286908078 |
| 0.323686 | 17 | 426 | 0.1431924882629108 | 0.10328638497652583 |
| 0.400000 | 5 | 454 | 0.09030837004405286 | 0.07929515418502203 |
| 0.500000 | 0 | 443 | 0.0744920993227991 | 0.0744920993227991 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003710 | -0.000130 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003566 | -0.000078 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002984 | -0.000157 |
| aux_account_months | zero | 21000 | 2461 | +0.001684 | -0.000301 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001343 | +0.000057 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000686 | -0.000057 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000109 | +0.000454 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000066 | -0.000022 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000131 | +0.000283 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000175 | +0.000013 |
| employment_type | self_employed | 20987 | 2895 | -0.000237 | +0.000074 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000251 | -0.000076 |
| employment_type | variable_income | 15540 | 2180 | -0.000508 | -0.000309 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000965 | +0.000140 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.001962 | -0.001353 |
| employment_type | salaried | 33473 | 4432 | -0.002629 | -0.000050 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002812 | +0.000090 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.006123 | -0.000260 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.