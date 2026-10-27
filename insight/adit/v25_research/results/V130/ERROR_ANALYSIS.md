# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -12 | 212 | 0.33490566037735847 | 0.3915094339622642 |
| 0.200000 | -4 | 345 | 0.22318840579710145 | 0.23478260869565218 |
| 0.323686 | -21 | 439 | 0.11617312072892938 | 0.16400911161731208 |
| 0.400000 | -18 | 480 | 0.0875 | 0.125 |
| 0.500000 | -9 | 510 | 0.054901960784313725 | 0.07254901960784314 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.007793 | +0.000346 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.002183 | -0.003417 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001112 | -0.001424 |
| employment_type | self_employed | 20987 | 2895 | +0.000524 | -0.000843 |
| employment_type | variable_income | 15540 | 2180 | +0.000188 | -0.001044 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000968 | -0.000976 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001159 | -0.000862 |
| employment_type | salaried | 33473 | 4432 | -0.001331 | -0.001038 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001434 | -0.001012 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001924 | -0.000955 |
| baseline_risk_band | under005 | 29853 | 535 | -0.002497 | -0.003664 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003368 | -0.001562 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003498 | -0.001072 |
| aux_account_months | zero | 21000 | 2461 | -0.003741 | -0.000984 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.003958 | -0.000550 |
| aux_allocation_count | zero | 23732 | 2895 | -0.004076 | -0.001032 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.010932 | -0.006739 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.010994 | -0.005945 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.