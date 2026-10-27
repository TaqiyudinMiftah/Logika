# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -8 | 319 | 0.3730407523510972 | 0.3981191222570533 |
| 0.200000 | -7 | 501 | 0.2155688622754491 | 0.22954091816367264 |
| 0.323686 | -51 | 625 | 0.0976 | 0.1792 |
| 0.400000 | -16 | 688 | 0.09447674418604651 | 0.11773255813953488 |
| 0.500000 | -13 | 697 | 0.058823529411764705 | 0.07747489239598278 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.010403 | -0.000692 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003280 | -0.003337 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002187 | -0.003679 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000307 | -0.000112 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000824 | -0.000645 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000855 | -0.000407 |
| employment_type | self_employed | 20987 | 2895 | -0.001203 | -0.000980 |
| aux_account_months | zero | 21000 | 2461 | -0.002169 | -0.000652 |
| employment_type | salaried | 33473 | 4432 | -0.002267 | -0.000379 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.003707 | -0.003395 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.003707 | -0.001028 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.004712 | -0.000655 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.005430 | -0.000681 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.005619 | -0.000669 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.005656 | -0.003836 |
| employment_type | variable_income | 15540 | 2180 | -0.005831 | -0.000904 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.006689 | -0.001220 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.007990 | -0.000821 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.