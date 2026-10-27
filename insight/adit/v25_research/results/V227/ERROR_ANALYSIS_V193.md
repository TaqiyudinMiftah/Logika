# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 247 | 0.3805668016194332 | 0.340080971659919 |
| 0.200000 | -7 | 370 | 0.22162162162162163 | 0.24054054054054055 |
| 0.323686 | 7 | 416 | 0.1346153846153846 | 0.11778846153846154 |
| 0.400000 | 8 | 462 | 0.09090909090909091 | 0.0735930735930736 |
| 0.500000 | 2 | 438 | 0.0730593607305936 | 0.0684931506849315 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003317 | +0.000021 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003037 | +0.000006 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.002280 | +0.000808 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001421 | -0.000134 |
| aux_account_months | zero | 21000 | 2461 | +0.001151 | -0.000148 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000543 | +0.000047 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000065 | -0.000015 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000421 | +0.000071 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000835 | -0.000069 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001087 | -0.000030 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001149 | -0.000005 |
| employment_type | self_employed | 20987 | 2895 | -0.001328 | +0.000051 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001453 | +0.000339 |
| employment_type | variable_income | 15540 | 2180 | -0.001763 | -0.000301 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.001868 | -0.001159 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002183 | +0.000015 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002218 | -0.000472 |
| employment_type | salaried | 33473 | 4432 | -0.002645 | +0.000000 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.