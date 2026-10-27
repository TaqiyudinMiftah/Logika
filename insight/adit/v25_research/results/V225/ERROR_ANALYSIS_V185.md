# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 11 | 244 | 0.38114754098360654 | 0.3360655737704918 |
| 0.200000 | 0 | 355 | 0.22816901408450704 | 0.22816901408450704 |
| 0.323686 | 12 | 422 | 0.13981042654028436 | 0.11137440758293839 |
| 0.400000 | 0 | 451 | 0.08425720620842572 | 0.08425720620842572 |
| 0.500000 | 3 | 442 | 0.0746606334841629 | 0.06787330316742081 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003516 | +0.000005 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003175 | +0.000031 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002041 | -0.000077 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.001730 | +0.001015 |
| aux_account_months | zero | 21000 | 2461 | +0.001518 | -0.000149 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000199 | -0.000000 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000056 | -0.000049 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000055 | +0.000071 |
| employment_type | variable_income | 15540 | 2180 | -0.000201 | -0.000283 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000271 | +0.000000 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000412 | +0.000008 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000489 | +0.000484 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000608 | +0.000031 |
| employment_type | self_employed | 20987 | 2895 | -0.000941 | +0.000083 |
| employment_type | salaried | 33473 | 4432 | -0.001987 | +0.000033 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002651 | +0.000091 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.003918 | -0.000901 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.006825 | -0.000384 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.