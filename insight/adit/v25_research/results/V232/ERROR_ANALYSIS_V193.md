# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 3 | 279 | 0.3942652329749104 | 0.3835125448028674 |
| 0.200000 | 10 | 416 | 0.25 | 0.22596153846153846 |
| 0.323686 | 7 | 513 | 0.13840155945419103 | 0.12475633528265107 |
| 0.400000 | -4 | 581 | 0.08433734939759036 | 0.09122203098106713 |
| 0.500000 | -4 | 557 | 0.06463195691202872 | 0.0718132854578097 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29938 | 544 | +0.004552 | -0.000807 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002438 | -0.000152 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002364 | -0.000100 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001569 | -0.000458 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001182 | -0.000261 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000701 | -0.000304 |
| aux_account_months | zero | 21000 | 2461 | -0.000305 | -0.000234 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000612 | -0.000363 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000923 | -0.000378 |
| employment_type | salaried | 33473 | 4432 | -0.000965 | -0.000366 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001011 | -0.000340 |
| employment_type | self_employed | 20987 | 2895 | -0.001120 | -0.000210 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001305 | -0.000357 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.001374 | +0.000079 |
| employment_type | variable_income | 15540 | 2180 | -0.001482 | -0.000434 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.003045 | -0.004013 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.004280 | -0.000687 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.007393 | -0.001907 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.