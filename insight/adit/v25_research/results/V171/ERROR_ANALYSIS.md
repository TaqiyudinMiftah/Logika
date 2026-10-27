# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -4 | 113 | 0.35398230088495575 | 0.3893805309734513 |
| 0.200000 | 10 | 201 | 0.24378109452736318 | 0.19402985074626866 |
| 0.323686 | -16 | 257 | 0.13229571984435798 | 0.19455252918287938 |
| 0.400000 | 7 | 321 | 0.10903426791277258 | 0.08722741433021806 |
| 0.500000 | 5 | 323 | 0.07739938080495357 | 0.06191950464396285 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.000902 | +0.000299 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000806 | +0.001168 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000678 | +0.000237 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000191 | +0.000176 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000019 | +0.000386 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000105 | +0.000992 |
| employment_type | salaried | 33473 | 4432 | -0.000245 | +0.000003 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000253 | +0.000461 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000425 | +0.000205 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000506 | +0.000002 |
| employment_type | self_employed | 20987 | 2895 | -0.000650 | +0.000324 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000950 | +0.001487 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001624 | -0.000778 |
| aux_account_months | zero | 21000 | 2461 | -0.002021 | -0.000016 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002032 | +0.000134 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002540 | -0.000004 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002682 | +0.000055 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003019 | -0.000239 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.