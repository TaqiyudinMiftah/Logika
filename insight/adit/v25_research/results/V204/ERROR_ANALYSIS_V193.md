# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -6 | 196 | 0.35714285714285715 | 0.3877551020408163 |
| 0.200000 | 14 | 297 | 0.23232323232323232 | 0.18518518518518517 |
| 0.323686 | 7 | 382 | 0.15706806282722513 | 0.1387434554973822 |
| 0.400000 | 15 | 397 | 0.11586901763224182 | 0.07808564231738035 |
| 0.500000 | -13 | 379 | 0.052770448548812667 | 0.0870712401055409 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6096 | 3493 | +0.003690 | -0.002363 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001796 | +0.000305 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001751 | +0.000088 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.001523 | -0.000794 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001206 | -0.000181 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001181 | -0.000264 |
| baseline_risk_band | under005 | 29938 | 544 | +0.001154 | +0.001256 |
| aux_account_months | zero | 21000 | 2461 | +0.000925 | -0.000295 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000436 | -0.000440 |
| employment_type | variable_income | 15540 | 2180 | +0.000430 | -0.000273 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000173 | +0.000072 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000308 | +0.000041 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000539 | -0.000284 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000706 | +0.000014 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000867 | +0.000528 |
| employment_type | salaried | 33473 | 4432 | -0.001100 | +0.000133 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001652 | -0.000118 |
| employment_type | self_employed | 20987 | 2895 | -0.001659 | -0.000210 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.