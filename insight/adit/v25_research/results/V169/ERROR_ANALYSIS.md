# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 108 | 0.35185185185185186 | 0.3425925925925926 |
| 0.200000 | 9 | 187 | 0.25133689839572193 | 0.20320855614973263 |
| 0.323686 | -16 | 259 | 0.11969111969111969 | 0.18146718146718147 |
| 0.400000 | 8 | 311 | 0.10932475884244373 | 0.08360128617363344 |
| 0.500000 | 0 | 337 | 0.0712166172106825 | 0.0712166172106825 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002498 | +0.001634 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000246 | -0.000978 |
| employment_type | variable_income | 15540 | 2180 | +0.000124 | +0.000384 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000422 | +0.000941 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000586 | +0.001611 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000732 | +0.000294 |
| employment_type | self_employed | 20987 | 2895 | -0.000757 | +0.000360 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000823 | +0.000103 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000902 | +0.000421 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000929 | +0.000281 |
| employment_type | salaried | 33473 | 4432 | -0.001151 | -0.000002 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001201 | +0.000255 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001348 | +0.000262 |
| aux_account_months | zero | 21000 | 2461 | -0.001626 | -0.000075 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001806 | -0.000051 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001865 | +0.000097 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002703 | -0.000034 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003913 | -0.000030 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.