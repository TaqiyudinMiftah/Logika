# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 17 | 211 | 0.41232227488151657 | 0.33175355450236965 |
| 0.200000 | 20 | 345 | 0.25217391304347825 | 0.19420289855072465 |
| 0.323686 | -16 | 443 | 0.12415349887133183 | 0.16027088036117382 |
| 0.400000 | -3 | 488 | 0.0942622950819672 | 0.10040983606557377 |
| 0.500000 | -16 | 518 | 0.05019305019305019 | 0.08108108108108109 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.018260 | +0.001187 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.007295 | +0.000843 |
| baseline_risk_band | under005 | 29853 | 535 | +0.004134 | +0.003410 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001660 | +0.000509 |
| employment_type | variable_income | 15540 | 2180 | +0.001330 | +0.000087 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.001042 | +0.000685 |
| employment_type | self_employed | 20987 | 2895 | +0.000915 | +0.000212 |
| employment_type | salaried | 33473 | 4432 | +0.000805 | +0.000448 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000299 | +0.000402 |
| aux_account_months | zero | 21000 | 2461 | +0.000255 | +0.000427 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000283 | +0.000280 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000368 | +0.000419 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000445 | +0.001534 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000644 | +0.000724 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001192 | +0.000306 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001225 | +0.000142 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001655 | +0.000348 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001910 | +0.000241 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001948 | +0.000082 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002276 | -0.001541 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002475 | +0.002537 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003583 | -0.000591 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.