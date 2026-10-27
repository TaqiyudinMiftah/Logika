# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -50 | 843 | 0.35231316725978645 | 0.41162514827995256 |
| 0.200000 | -37 | 1295 | 0.21081081081081082 | 0.23938223938223938 |
| 0.323686 | -89 | 1652 | 0.11319612590799032 | 0.16707021791767554 |
| 0.400000 | -34 | 1869 | 0.09844836811128946 | 0.11663991439272338 |
| 0.500000 | -45 | 1988 | 0.060362173038229376 | 0.0829979879275654 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.006012 | -0.004422 |
| aux_account_months | zero | 21000 | 2461 | -0.006830 | -0.003460 |
| employment_type | salaried | 33473 | 4432 | -0.006891 | -0.004118 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.007850 | -0.003379 |
| employment_type | variable_income | 15540 | 2180 | -0.007911 | -0.004745 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.007917 | -0.003762 |
| aux_allocation_count | zero | 23732 | 2895 | -0.008228 | -0.003553 |
| employment_type | self_employed | 20987 | 2895 | -0.009601 | -0.004198 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.009852 | -0.004663 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.009927 | -0.004726 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.010129 | -0.004691 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.011364 | -0.014978 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.012925 | -0.005468 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.017538 | -0.005634 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.018341 | -0.005232 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.024299 | -0.015593 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.025378 | -0.016141 |
| baseline_risk_band | under005 | 29853 | 535 | -0.027765 | -0.017158 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.