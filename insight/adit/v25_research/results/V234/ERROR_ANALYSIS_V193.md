# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 246 | 0.3699186991869919 | 0.34146341463414637 |
| 0.200000 | -2 | 376 | 0.22872340425531915 | 0.23404255319148937 |
| 0.323686 | 8 | 427 | 0.1405152224824356 | 0.12177985948477751 |
| 0.400000 | 10 | 460 | 0.09565217391304348 | 0.07391304347826087 |
| 0.500000 | 0 | 440 | 0.07045454545454545 | 0.07045454545454545 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003296 | -0.000049 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002969 | -0.000073 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001750 | -0.000206 |
| aux_account_months | zero | 21000 | 2461 | +0.001579 | -0.000239 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001567 | +0.000022 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000059 | -0.000060 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000387 | +0.000381 |
| employment_type | self_employed | 20987 | 2895 | -0.000416 | +0.000005 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000871 | -0.000099 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000891 | +0.000122 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000947 | -0.000066 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000964 | -0.000036 |
| baseline_risk_band | under005 | 29938 | 544 | -0.001120 | -0.000044 |
| employment_type | variable_income | 15540 | 2180 | -0.001305 | -0.000335 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001876 | -0.000034 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002882 | -0.000367 |
| employment_type | salaried | 33473 | 4432 | -0.002902 | -0.000067 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.003277 | -0.001545 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.