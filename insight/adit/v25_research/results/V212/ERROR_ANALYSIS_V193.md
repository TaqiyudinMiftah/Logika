# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 32 | 0.46875 | 0.40625 |
| 0.200000 | -7 | 57 | 0.24561403508771928 | 0.3684210526315789 |
| 0.323686 | -3 | 63 | 0.15873015873015872 | 0.20634920634920634 |
| 0.400000 | 5 | 68 | 0.11764705882352941 | 0.04411764705882353 |
| 0.500000 | 1 | 69 | 0.08695652173913043 | 0.07246376811594203 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.003256 | -0.000249 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001270 | -0.000312 |
| employment_type | self_employed | 20987 | 2895 | +0.001202 | +0.000054 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001196 | +0.000088 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001126 | +0.000065 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000602 | +0.000008 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.000461 | +0.000224 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000312 | -0.000052 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000184 | +0.000003 |
| employment_type | salaried | 33473 | 4432 | -0.000039 | -0.000025 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000054 | +0.000009 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000074 | -0.000016 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000308 | +0.000261 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000434 | +0.000079 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000562 | -0.000008 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000626 | -0.000006 |
| aux_account_months | zero | 21000 | 2461 | -0.000819 | +0.000068 |
| employment_type | variable_income | 15540 | 2180 | -0.000977 | -0.000003 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.