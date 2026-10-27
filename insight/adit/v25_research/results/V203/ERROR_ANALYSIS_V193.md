# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 77 | 0.4155844155844156 | 0.33766233766233766 |
| 0.200000 | -6 | 138 | 0.2318840579710145 | 0.2753623188405797 |
| 0.323686 | 0 | 156 | 0.14743589743589744 | 0.14743589743589744 |
| 0.400000 | 11 | 158 | 0.10126582278481013 | 0.03164556962025317 |
| 0.500000 | 1 | 163 | 0.06748466257668712 | 0.06134969325153374 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29938 | 544 | +0.003336 | +0.001528 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002427 | -0.000019 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002247 | +0.000045 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001440 | +0.000063 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001204 | +0.000074 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001074 | +0.000104 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000910 | -0.000067 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000801 | -0.000789 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.000740 | +0.000069 |
| aux_account_months | zero | 21000 | 2461 | +0.000676 | +0.000069 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000598 | +0.000151 |
| employment_type | variable_income | 15540 | 2180 | -0.000310 | +0.000066 |
| employment_type | salaried | 33473 | 4432 | -0.000613 | +0.000050 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000724 | +0.000113 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001110 | +0.000038 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001172 | +0.000028 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001306 | +0.000033 |
| employment_type | self_employed | 20987 | 2895 | -0.001907 | +0.000026 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.