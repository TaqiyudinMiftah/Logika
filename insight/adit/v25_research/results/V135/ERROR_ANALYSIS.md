# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -6 | 136 | 0.38235294117647056 | 0.4264705882352941 |
| 0.200000 | 0 | 201 | 0.24875621890547264 | 0.24875621890547264 |
| 0.323686 | -18 | 244 | 0.08196721311475409 | 0.1557377049180328 |
| 0.400000 | 3 | 273 | 0.08424908424908426 | 0.07326007326007326 |
| 0.500000 | 4 | 283 | 0.07773851590106007 | 0.0636042402826855 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.000959 | +0.000049 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000553 | -0.002144 |
| employment_type | salaried | 33473 | 4432 | +0.000478 | -0.000156 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000189 | -0.001151 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000032 | -0.000105 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000112 | -0.002638 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000126 | -0.000152 |
| aux_account_months | zero | 21000 | 2461 | -0.000342 | -0.000094 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000437 | -0.000116 |
| employment_type | self_employed | 20987 | 2895 | -0.000439 | -0.000228 |
| employment_type | variable_income | 15540 | 2180 | -0.001009 | -0.000405 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001767 | -0.000326 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001809 | -0.000311 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001835 | -0.000289 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.002230 | -0.000163 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002693 | -0.001325 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002785 | -0.000053 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.005428 | -0.000578 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.