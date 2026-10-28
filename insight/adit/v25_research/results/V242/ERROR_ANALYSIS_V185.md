# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 11 | 270 | 0.3851851851851852 | 0.34444444444444444 |
| 0.200000 | -17 | 433 | 0.20092378752886836 | 0.24018475750577367 |
| 0.323686 | 10 | 514 | 0.1264591439688716 | 0.10700389105058365 |
| 0.400000 | -2 | 558 | 0.08064516129032258 | 0.08422939068100359 |
| 0.500000 | -7 | 565 | 0.06371681415929203 | 0.07610619469026549 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.004225 | -0.000010 |
| baseline_risk_band | under005 | 29965 | 546 | +0.003920 | +0.002088 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003845 | -0.000122 |
| aux_account_months | zero | 21000 | 2461 | +0.002183 | -0.000255 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002168 | -0.000065 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.001749 | -0.000027 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001706 | +0.000165 |
| employment_type | variable_income | 15540 | 2180 | -0.000614 | -0.000488 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001188 | -0.000038 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001258 | +0.000074 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001281 | +0.000080 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001457 | +0.000583 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.001818 | +0.000592 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002218 | -0.000976 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002261 | -0.000105 |
| employment_type | self_employed | 20987 | 2895 | -0.002448 | +0.000126 |
| employment_type | salaried | 33473 | 4432 | -0.002760 | +0.000125 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004702 | -0.000262 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.