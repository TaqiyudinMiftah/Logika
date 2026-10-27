# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 90 | 0.43333333333333335 | 0.4222222222222222 |
| 0.200000 | -8 | 163 | 0.2085889570552147 | 0.25766871165644173 |
| 0.323686 | 6 | 196 | 0.15816326530612246 | 0.12755102040816327 |
| 0.400000 | 10 | 192 | 0.109375 | 0.057291666666666664 |
| 0.500000 | 0 | 192 | 0.06770833333333333 | 0.06770833333333333 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.005309 | +0.000216 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002651 | +0.000013 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.002568 | -0.000540 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.002070 | -0.000128 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001101 | -0.000059 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000794 | -0.000105 |
| employment_type | variable_income | 15540 | 2180 | +0.000658 | -0.000050 |
| baseline_risk_band | under005 | 29938 | 544 | +0.000656 | +0.000048 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000472 | -0.000386 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000431 | -0.000115 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000206 | -0.001502 |
| aux_account_months | zero | 21000 | 2461 | -0.000309 | -0.000171 |
| employment_type | salaried | 33473 | 4432 | -0.001211 | -0.000111 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001257 | -0.000039 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001314 | -0.000039 |
| employment_type | self_employed | 20987 | 2895 | -0.001368 | -0.000059 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001413 | -0.000056 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001557 | -0.000070 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.