# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 261 | 0.3831417624521073 | 0.36398467432950193 |
| 0.200000 | 1 | 404 | 0.22524752475247525 | 0.22277227722772278 |
| 0.323686 | 1 | 486 | 0.13168724279835392 | 0.12962962962962962 |
| 0.400000 | -3 | 525 | 0.0838095238095238 | 0.08952380952380952 |
| 0.500000 | 4 | 514 | 0.07587548638132295 | 0.06809338521400778 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003768 | -0.000352 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003645 | -0.000316 |
| aux_account_months | zero | 21000 | 2461 | +0.001407 | -0.000465 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000429 | +0.000345 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000121 | -0.000271 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000531 | -0.000088 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000570 | -0.000074 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001103 | -0.000625 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001170 | -0.000147 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001662 | -0.000133 |
| employment_type | variable_income | 15540 | 2180 | -0.001696 | -0.000399 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001777 | -0.000093 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001838 | -0.000886 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.001958 | -0.001446 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002052 | -0.000099 |
| employment_type | self_employed | 20987 | 2895 | -0.002406 | -0.000190 |
| employment_type | salaried | 33473 | 4432 | -0.002973 | -0.000130 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005232 | -0.000788 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.