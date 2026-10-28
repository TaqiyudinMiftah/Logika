# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 245 | 0.37142857142857144 | 0.3510204081632653 |
| 0.200000 | -2 | 382 | 0.22251308900523561 | 0.22774869109947643 |
| 0.323686 | 12 | 443 | 0.13318284424379231 | 0.10609480812641084 |
| 0.400000 | 4 | 477 | 0.09014675052410902 | 0.08176100628930817 |
| 0.500000 | 3 | 464 | 0.07758620689655173 | 0.07112068965517242 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003243 | -0.000228 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003097 | -0.000197 |
| aux_account_months | zero | 21000 | 2461 | +0.001493 | -0.000364 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.001003 | -0.001421 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000808 | -0.000027 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000323 | -0.000134 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000266 | -0.000172 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000144 | -0.000605 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000196 | -0.000107 |
| employment_type | variable_income | 15540 | 2180 | -0.000332 | -0.000376 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001002 | -0.000105 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001019 | -0.000059 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001168 | -0.000075 |
| employment_type | self_employed | 20987 | 2895 | -0.001301 | -0.000014 |
| employment_type | salaried | 33473 | 4432 | -0.001716 | -0.000119 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.002008 | +0.000103 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003372 | -0.000075 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005915 | -0.000583 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.