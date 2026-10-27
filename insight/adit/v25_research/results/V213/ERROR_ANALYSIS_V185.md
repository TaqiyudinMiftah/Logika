# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 35 | 0.42857142857142855 | 0.4 |
| 0.200000 | 1 | 51 | 0.29411764705882354 | 0.27450980392156865 |
| 0.323686 | 3 | 75 | 0.16 | 0.12 |
| 0.400000 | 3 | 56 | 0.05357142857142857 | 0.0 |
| 0.500000 | 0 | 73 | 0.0684931506849315 | 0.0684931506849315 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.004774 | +0.000443 |
| baseline_risk_band | under005 | 29965 | 546 | +0.002207 | +0.000550 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.001027 | +0.000566 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000516 | +0.000182 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000479 | +0.000086 |
| employment_type | self_employed | 20987 | 2895 | +0.000442 | +0.000095 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000354 | +0.000075 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000208 | +0.000108 |
| aux_account_months | zero | 21000 | 2461 | +0.000173 | +0.000052 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000134 | +0.000082 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000122 | +0.000100 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000024 | +0.000104 |
| employment_type | variable_income | 15540 | 2180 | -0.000106 | +0.000037 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000316 | +0.000110 |
| employment_type | salaried | 33473 | 4432 | -0.000539 | +0.000105 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000553 | +0.000078 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000905 | +0.000647 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002369 | -0.000044 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.