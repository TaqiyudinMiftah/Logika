# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 26 | 0.3076923076923077 | 0.34615384615384615 |
| 0.200000 | 2 | 43 | 0.27906976744186046 | 0.23255813953488372 |
| 0.323686 | 0 | 57 | 0.10526315789473684 | 0.10526315789473684 |
| 0.400000 | 2 | 46 | 0.06521739130434782 | 0.021739130434782608 |
| 0.500000 | -1 | 64 | 0.03125 | 0.046875 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.002308 | +0.000150 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000941 | +0.000067 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000788 | +0.000062 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000692 | +0.000026 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000661 | +0.000366 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000637 | +0.000060 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000603 | +0.000041 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000588 | +0.000019 |
| aux_account_months | zero | 21000 | 2461 | +0.000558 | +0.000013 |
| employment_type | variable_income | 15540 | 2180 | +0.000541 | +0.000000 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000489 | +0.000058 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000465 | +0.000026 |
| employment_type | self_employed | 20987 | 2895 | +0.000069 | +0.000069 |
| employment_type | salaried | 33473 | 4432 | +0.000065 | +0.000046 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000051 | +0.000086 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000374 | +0.000378 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000970 | +0.000094 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002369 | -0.000080 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.