# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 277 | 0.3898916967509025 | 0.38267148014440433 |
| 0.200000 | 10 | 413 | 0.24455205811138014 | 0.22033898305084745 |
| 0.323686 | 12 | 518 | 0.14478764478764478 | 0.12162162162162163 |
| 0.400000 | -7 | 578 | 0.07958477508650519 | 0.09169550173010381 |
| 0.500000 | -4 | 561 | 0.0659536541889483 | 0.07308377896613191 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29965 | 546 | +0.003706 | -0.000835 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002875 | -0.000122 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002669 | -0.000168 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001623 | -0.000440 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000958 | -0.000259 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000478 | -0.000289 |
| aux_account_months | zero | 21000 | 2461 | +0.000137 | -0.000265 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000089 | -0.000355 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000381 | -0.000379 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000519 | -0.000339 |
| employment_type | self_employed | 20987 | 2895 | -0.000760 | -0.000209 |
| employment_type | salaried | 33473 | 4432 | -0.000838 | -0.000370 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000974 | -0.000366 |
| employment_type | variable_income | 15540 | 2180 | -0.001173 | -0.000428 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.001490 | +0.000165 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.003916 | -0.000688 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.004214 | -0.004074 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.009554 | -0.001807 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.