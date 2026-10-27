# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -14 | 264 | 0.32575757575757575 | 0.3787878787878788 |
| 0.200000 | 18 | 402 | 0.2537313432835821 | 0.208955223880597 |
| 0.323686 | -5 | 490 | 0.11224489795918367 | 0.12244897959183673 |
| 0.400000 | 8 | 528 | 0.09659090909090909 | 0.08143939393939394 |
| 0.500000 | -15 | 524 | 0.061068702290076333 | 0.08969465648854962 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.004398 | -0.000113 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.004347 | +0.000036 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.004114 | -0.002117 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.003993 | +0.000060 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.002042 | -0.000100 |
| employment_type | variable_income | 15540 | 2180 | +0.001862 | +0.000067 |
| employment_type | salaried | 33473 | 4432 | +0.001588 | +0.000173 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001514 | +0.002723 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000516 | +0.000101 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000510 | +0.000040 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000387 | +0.000091 |
| employment_type | self_employed | 20987 | 2895 | -0.000414 | -0.000257 |
| aux_account_months | zero | 21000 | 2461 | -0.000478 | -0.000017 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000732 | +0.000464 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001199 | +0.000130 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001352 | +0.000355 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001711 | -0.000154 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001825 | -0.000063 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001945 | -0.000117 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.002085 | -0.000307 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003396 | -0.000226 |
| orders90_exposure_band | zero | 3428 | 106 | -0.012365 | -0.000545 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.