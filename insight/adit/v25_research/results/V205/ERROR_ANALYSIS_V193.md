# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 272 | 0.3786764705882353 | 0.3713235294117647 |
| 0.200000 | 23 | 455 | 0.24395604395604395 | 0.1934065934065934 |
| 0.323686 | 0 | 534 | 0.11797752808988764 | 0.11797752808988764 |
| 0.400000 | 3 | 567 | 0.09347442680776014 | 0.08818342151675485 |
| 0.500000 | -10 | 529 | 0.06427221172022685 | 0.0831758034026465 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29938 | 544 | +0.009101 | +0.002192 |
| employment_type | variable_income | 15540 | 2180 | +0.002363 | -0.000388 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.002264 | +0.001022 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002249 | +0.000002 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002173 | -0.000003 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001394 | +0.000080 |
| aux_account_months | zero | 21000 | 2461 | +0.001209 | -0.000135 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000926 | -0.000957 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000455 | -0.000996 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000428 | +0.000040 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000430 | +0.000171 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000483 | +0.001684 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001526 | +0.000167 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001676 | +0.000072 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001781 | +0.000145 |
| employment_type | self_employed | 20987 | 2895 | -0.002048 | +0.000174 |
| employment_type | salaried | 33473 | 4432 | -0.002417 | +0.000217 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.005877 | +0.000294 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.