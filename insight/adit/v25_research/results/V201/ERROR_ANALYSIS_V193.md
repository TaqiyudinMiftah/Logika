# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 248 | 0.375 | 0.34274193548387094 |
| 0.200000 | -4 | 377 | 0.22015915119363394 | 0.23076923076923078 |
| 0.323686 | 13 | 434 | 0.1336405529953917 | 0.10368663594470046 |
| 0.400000 | 8 | 460 | 0.09565217391304348 | 0.0782608695652174 |
| 0.500000 | 0 | 442 | 0.07239819004524888 | 0.07239819004524888 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.003549 | -0.000139 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003065 | -0.000055 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002950 | -0.000105 |
| baseline_risk_band | under005 | 29938 | 544 | +0.001299 | +0.000058 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001006 | +0.000040 |
| aux_account_months | zero | 21000 | 2461 | +0.000997 | -0.000274 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.000495 | +0.000343 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000183 | -0.001275 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000038 | -0.000041 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000368 | -0.000016 |
| employment_type | variable_income | 15540 | 2180 | -0.000462 | -0.000308 |
| employment_type | self_employed | 20987 | 2895 | -0.000513 | +0.000083 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000558 | +0.000015 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000604 | -0.000073 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.001750 | +0.000391 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002180 | -0.000338 |
| employment_type | salaried | 33473 | 4432 | -0.002725 | -0.000049 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002953 | +0.000092 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.