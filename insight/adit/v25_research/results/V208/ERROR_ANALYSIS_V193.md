# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -5 | 305 | 0.35737704918032787 | 0.3737704918032787 |
| 0.200000 | -1 | 480 | 0.24583333333333332 | 0.24791666666666667 |
| 0.323686 | 1 | 598 | 0.13712374581939799 | 0.1354515050167224 |
| 0.400000 | 1 | 667 | 0.10194902548725637 | 0.10044977511244378 |
| 0.500000 | -17 | 703 | 0.06401137980085349 | 0.08819345661450925 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.004840 | -0.001008 |
| baseline_risk_band | under005 | 29938 | 544 | +0.001787 | -0.002229 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000174 | -0.007232 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000158 | -0.000588 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000341 | -0.000392 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001459 | -0.000947 |
| employment_type | salaried | 33473 | 4432 | -0.002044 | -0.001003 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002065 | -0.000927 |
| employment_type | variable_income | 15540 | 2180 | -0.002253 | -0.000570 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002301 | -0.000936 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002467 | -0.001028 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.003039 | -0.003350 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003433 | -0.000510 |
| aux_account_months | zero | 21000 | 2461 | -0.003927 | -0.000405 |
| employment_type | self_employed | 20987 | 2895 | -0.004380 | -0.000875 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004982 | -0.001298 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005270 | -0.003446 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.010813 | -0.003311 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.