# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -5 | 48 | 0.3125 | 0.4166666666666667 |
| 0.200000 | 3 | 63 | 0.25396825396825395 | 0.20634920634920634 |
| 0.323686 | -10 | 81 | 0.06172839506172839 | 0.18518518518518517 |
| 0.400000 | 0 | 86 | 0.11627906976744186 | 0.11627906976744186 |
| 0.500000 | 4 | 87 | 0.10344827586206896 | 0.05747126436781609 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001810 | -0.000074 |
| employment_type | self_employed | 20987 | 2895 | +0.000679 | -0.000082 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000397 | -0.000133 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000146 | -0.000179 |
| employment_type | salaried | 33473 | 4432 | +0.000040 | -0.000002 |
| aux_account_months | zero | 21000 | 2461 | -0.000054 | +0.000003 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000059 | +0.000005 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000201 | -0.000243 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000489 | +0.000008 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000627 | -0.000125 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000705 | +0.000036 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000796 | -0.000019 |
| employment_type | variable_income | 15540 | 2180 | -0.000838 | -0.000044 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000839 | -0.000363 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000915 | -0.000019 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001012 | -0.000043 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002027 | +0.000042 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002712 | -0.000975 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.