# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -5 | 127 | 0.36220472440944884 | 0.4015748031496063 |
| 0.200000 | 5 | 195 | 0.2717948717948718 | 0.24615384615384617 |
| 0.323686 | -21 | 235 | 0.08936170212765958 | 0.17872340425531916 |
| 0.400000 | 1 | 258 | 0.08139534883720931 | 0.07751937984496124 |
| 0.500000 | -2 | 262 | 0.06870229007633588 | 0.07633587786259542 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.002783 | -0.001243 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001516 | -0.000941 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001122 | -0.001029 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000747 | +0.000124 |
| aux_account_months | zero | 21000 | 2461 | +0.000463 | -0.000034 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000417 | +0.000012 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000202 | +0.000002 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000175 | -0.001883 |
| employment_type | salaried | 33473 | 4432 | -0.000100 | -0.000043 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000793 | -0.000016 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001285 | +0.000330 |
| employment_type | variable_income | 15540 | 2180 | -0.001290 | -0.000340 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001290 | -0.000090 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001351 | -0.000179 |
| employment_type | self_employed | 20987 | 2895 | -0.001421 | -0.000132 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001615 | -0.000222 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002324 | -0.000228 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.005082 | -0.000461 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.