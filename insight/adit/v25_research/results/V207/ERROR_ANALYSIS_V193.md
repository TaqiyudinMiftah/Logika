# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 374 | 0.37967914438502676 | 0.3663101604278075 |
| 0.200000 | 4 | 579 | 0.24006908462867013 | 0.23316062176165803 |
| 0.323686 | -6 | 713 | 0.11921458625525946 | 0.1276297335203366 |
| 0.400000 | 3 | 806 | 0.09801488833746898 | 0.09429280397022333 |
| 0.500000 | -18 | 827 | 0.0592503022974607 | 0.08101571946795647 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.003671 | -0.000887 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000398 | -0.002587 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000582 | -0.000567 |
| employment_type | salaried | 33473 | 4432 | -0.000855 | -0.000960 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001413 | -0.000995 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002053 | -0.000401 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002188 | -0.000655 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002212 | -0.000934 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002225 | -0.000952 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.002347 | -0.001326 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002450 | -0.001017 |
| employment_type | self_employed | 20987 | 2895 | -0.002889 | -0.000802 |
| employment_type | variable_income | 15540 | 2180 | -0.003175 | -0.000851 |
| aux_account_months | zero | 21000 | 2461 | -0.004164 | -0.000513 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004269 | -0.001345 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.006001 | -0.008286 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.011161 | -0.002791 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.012493 | -0.004199 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.