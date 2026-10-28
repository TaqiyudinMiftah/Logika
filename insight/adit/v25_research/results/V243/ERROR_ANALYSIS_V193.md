# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 12 | 255 | 0.39215686274509803 | 0.34509803921568627 |
| 0.200000 | 0 | 399 | 0.22807017543859648 | 0.22807017543859648 |
| 0.323686 | 1 | 469 | 0.12366737739872068 | 0.12153518123667377 |
| 0.400000 | 3 | 503 | 0.09145129224652088 | 0.08548707753479125 |
| 0.500000 | -3 | 480 | 0.06666666666666667 | 0.07291666666666667 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6096 | 3493 | +0.004487 | +0.000723 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002866 | -0.000165 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002451 | -0.000182 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000423 | -0.000132 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000217 | -0.001834 |
| aux_account_months | zero | 21000 | 2461 | +0.000130 | -0.000350 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000592 | -0.000261 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000779 | -0.000294 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000850 | -0.000144 |
| employment_type | variable_income | 15540 | 2180 | -0.001099 | -0.000394 |
| baseline_risk_band | under005 | 29938 | 544 | -0.001625 | -0.000780 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001779 | -0.000022 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002138 | -0.000118 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002345 | -0.000137 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002407 | -0.000185 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002408 | -0.000765 |
| employment_type | self_employed | 20987 | 2895 | -0.002682 | -0.000153 |
| employment_type | salaried | 33473 | 4432 | -0.003851 | -0.000117 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.