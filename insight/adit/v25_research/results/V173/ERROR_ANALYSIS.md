# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -3 | 154 | 0.34415584415584416 | 0.36363636363636365 |
| 0.200000 | 8 | 243 | 0.24691358024691357 | 0.2139917695473251 |
| 0.323686 | -20 | 340 | 0.11764705882352941 | 0.17647058823529413 |
| 0.400000 | 12 | 389 | 0.11311053984575835 | 0.08226221079691516 |
| 0.500000 | -7 | 394 | 0.050761421319796954 | 0.06852791878172589 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003962 | +0.000350 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001499 | +0.000721 |
| employment_type | variable_income | 15540 | 2180 | +0.001239 | +0.000352 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000234 | +0.001571 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000061 | -0.002531 |
| employment_type | self_employed | 20987 | 2895 | -0.000014 | +0.000170 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000467 | +0.000199 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000580 | +0.000263 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000774 | +0.000105 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000910 | -0.000149 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000994 | +0.000113 |
| employment_type | salaried | 33473 | 4432 | -0.001148 | -0.000236 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002380 | +0.000677 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002437 | -0.000066 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002444 | -0.000134 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.004019 | -0.000289 |
| aux_allocation_count | zero | 23732 | 2895 | -0.004146 | -0.000249 |
| aux_account_months | zero | 21000 | 2461 | -0.004305 | -0.000262 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.