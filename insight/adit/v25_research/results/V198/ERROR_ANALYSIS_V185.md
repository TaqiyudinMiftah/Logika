# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 209 | 0.37320574162679426 | 0.35406698564593303 |
| 0.200000 | 19 | 337 | 0.26409495548961426 | 0.20771513353115728 |
| 0.323686 | 5 | 391 | 0.13043478260869565 | 0.11764705882352941 |
| 0.400000 | 11 | 414 | 0.09903381642512077 | 0.07246376811594203 |
| 0.500000 | -9 | 423 | 0.0591016548463357 | 0.08037825059101655 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29965 | 546 | +0.009252 | +0.002751 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.003840 | +0.001377 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.003568 | -0.000229 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003375 | -0.000388 |
| aux_account_months | zero | 21000 | 2461 | +0.001791 | +0.000164 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001426 | +0.000183 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001261 | -0.000047 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000842 | +0.000146 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000021 | +0.000231 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000888 | +0.000237 |
| employment_type | variable_income | 15540 | 2180 | -0.001011 | -0.000222 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.001236 | +0.000465 |
| employment_type | self_employed | 20987 | 2895 | -0.001481 | +0.000115 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001506 | +0.000085 |
| employment_type | salaried | 33473 | 4432 | -0.001572 | +0.000343 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001989 | +0.000176 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002007 | +0.000156 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.004078 | +0.000524 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.