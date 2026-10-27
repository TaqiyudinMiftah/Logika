# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 10 | 0.5 | 0.3 |
| 0.200000 | -2 | 17 | 0.23529411764705882 | 0.35294117647058826 |
| 0.323686 | -5 | 30 | 0.06666666666666667 | 0.23333333333333334 |
| 0.400000 | 1 | 24 | 0.08333333333333333 | 0.041666666666666664 |
| 0.500000 | -1 | 27 | 0.0 | 0.037037037037037035 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.001282 | -0.000006 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000998 | +0.000179 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000672 | +0.000005 |
| employment_type | self_employed | 20987 | 2895 | +0.000401 | +0.000017 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000170 | -0.000078 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000146 | +0.000015 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000146 | +0.000020 |
| aux_account_months | zero | 21000 | 2461 | +0.000144 | +0.000019 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000135 | +0.000018 |
| employment_type | salaried | 33473 | 4432 | +0.000072 | +0.000025 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000051 | +0.000014 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000036 | +0.000023 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000015 | +0.000011 |
| employment_type | variable_income | 15540 | 2180 | -0.000152 | -0.000018 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000194 | +0.000021 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000366 | +0.000019 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000374 | +0.000179 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000536 | -0.000014 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.