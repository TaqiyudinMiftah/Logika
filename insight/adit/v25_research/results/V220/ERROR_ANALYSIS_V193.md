# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 22 | 0.4090909090909091 | 0.4090909090909091 |
| 0.200000 | 2 | 40 | 0.275 | 0.225 |
| 0.323686 | -5 | 57 | 0.08771929824561403 | 0.17543859649122806 |
| 0.400000 | 5 | 43 | 0.13953488372093023 | 0.023255813953488372 |
| 0.500000 | -1 | 61 | 0.01639344262295082 | 0.03278688524590164 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.003651 | +0.000176 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000887 | +0.000049 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000690 | +0.000024 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000556 | +0.000370 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000357 | +0.000036 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000295 | +0.000061 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000272 | +0.000050 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.000267 | +0.000365 |
| employment_type | variable_income | 15540 | 2180 | +0.000232 | -0.000006 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000180 | +0.000049 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000173 | +0.000071 |
| aux_account_months | zero | 21000 | 2461 | +0.000115 | +0.000044 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000095 | +0.000060 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000034 | +0.000050 |
| employment_type | salaried | 33473 | 4432 | -0.000062 | +0.000050 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000208 | -0.000180 |
| employment_type | self_employed | 20987 | 2895 | -0.000290 | +0.000068 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000996 | +0.000098 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.