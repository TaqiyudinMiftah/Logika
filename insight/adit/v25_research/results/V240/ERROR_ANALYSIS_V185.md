# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 308 | 0.36688311688311687 | 0.35064935064935066 |
| 0.200000 | 7 | 469 | 0.21961620469083157 | 0.2046908315565032 |
| 0.323686 | -5 | 579 | 0.11917098445595854 | 0.12780656303972365 |
| 0.400000 | -6 | 595 | 0.08067226890756303 | 0.0907563025210084 |
| 0.500000 | -6 | 587 | 0.059625212947189095 | 0.06984667802385008 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.004018 | +0.000103 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003568 | -0.000112 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.003039 | -0.001052 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.002547 | +0.000391 |
| baseline_risk_band | under005 | 29965 | 546 | +0.002413 | +0.001625 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001548 | +0.000089 |
| aux_account_months | zero | 21000 | 2461 | +0.001529 | -0.000295 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000501 | +0.001498 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000070 | +0.000088 |
| employment_type | variable_income | 15540 | 2180 | -0.000644 | -0.000436 |
| employment_type | self_employed | 20987 | 2895 | -0.000856 | +0.000011 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001303 | -0.000071 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001692 | +0.000053 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.002055 | -0.000109 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002349 | +0.000094 |
| employment_type | salaried | 33473 | 4432 | -0.003136 | +0.000178 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004093 | +0.000043 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004132 | +0.000069 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.