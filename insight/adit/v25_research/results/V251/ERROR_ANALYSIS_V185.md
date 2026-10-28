# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 12 | 244 | 0.38934426229508196 | 0.3401639344262295 |
| 0.200000 | 1 | 370 | 0.22432432432432434 | 0.22162162162162163 |
| 0.323686 | 9 | 444 | 0.13063063063063063 | 0.11036036036036036 |
| 0.400000 | 4 | 474 | 0.08438818565400844 | 0.0759493670886076 |
| 0.500000 | 0 | 455 | 0.07032967032967033 | 0.07032967032967033 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003028 | -0.000123 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002849 | -0.000163 |
| aux_account_months | zero | 21000 | 2461 | +0.001360 | -0.000345 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.001344 | +0.000610 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001108 | -0.000028 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000739 | -0.000212 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000062 | -0.000102 |
| employment_type | variable_income | 15540 | 2180 | -0.000182 | -0.000363 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000803 | -0.000043 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000810 | -0.000012 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000882 | -0.000075 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001174 | -0.000123 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.001304 | -0.001520 |
| employment_type | self_employed | 20987 | 2895 | -0.001715 | -0.000037 |
| baseline_risk_band | under005 | 29965 | 546 | -0.002006 | -0.000515 |
| employment_type | salaried | 33473 | 4432 | -0.002723 | -0.000082 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002758 | +0.000030 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005460 | -0.000444 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.