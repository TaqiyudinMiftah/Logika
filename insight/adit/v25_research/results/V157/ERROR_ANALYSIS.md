# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 13 | 0.5384615384615384 | 0.46153846153846156 |
| 0.200000 | 1 | 21 | 0.14285714285714285 | 0.09523809523809523 |
| 0.323686 | -4 | 22 | 0.045454545454545456 | 0.22727272727272727 |
| 0.400000 | 4 | 15 | 0.3333333333333333 | 0.06666666666666667 |
| 0.500000 | 2 | 23 | 0.08695652173913043 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | self_employed | 20987 | 2895 | +0.000276 | -0.000027 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000228 | -0.000166 |
| employment_type | variable_income | 15540 | 2180 | +0.000215 | -0.000031 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000136 | -0.000112 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000054 | -0.000043 |
| employment_type | salaried | 33473 | 4432 | +0.000033 | -0.000039 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000008 | -0.000034 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000073 | -0.000031 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000076 | -0.000026 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000196 | -0.000037 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000253 | -0.000037 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000339 | -0.000033 |
| aux_account_months | zero | 21000 | 2461 | -0.000356 | -0.000019 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000544 | -0.000057 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000653 | -0.000013 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001249 | -0.000201 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001612 | +0.000002 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.001923 | -0.000248 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.