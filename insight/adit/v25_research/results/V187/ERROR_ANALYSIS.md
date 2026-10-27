# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -4 | 21 | 0.2857142857142857 | 0.47619047619047616 |
| 0.200000 | 0 | 33 | 0.18181818181818182 | 0.18181818181818182 |
| 0.323686 | -6 | 38 | 0.15789473684210525 | 0.3157894736842105 |
| 0.400000 | 1 | 35 | 0.17142857142857143 | 0.14285714285714285 |
| 0.500000 | 0 | 49 | 0.02040816326530612 | 0.02040816326530612 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.001484 | -0.000027 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001145 | -0.000028 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001080 | -0.000054 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.000575 | +0.000056 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000487 | +0.000015 |
| employment_type | salaried | 33473 | 4432 | +0.000264 | -0.000020 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000057 | -0.000019 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000009 | +0.000016 |
| aux_account_months | zero | 21000 | 2461 | -0.000025 | +0.000028 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000100 | +0.000011 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000100 | -0.000027 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000176 | +0.000003 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000207 | -0.000063 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000367 | -0.000026 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000609 | -0.000033 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000653 | -0.000030 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.000672 | -0.000124 |
| employment_type | self_employed | 20987 | 2895 | -0.000705 | -0.000006 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000772 | -0.000097 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001074 | -0.000030 |
| orders90_exposure_band | zero | 3428 | 106 | -0.001204 | -0.000335 |
| baseline_risk_band | under005 | 29853 | 535 | -0.002435 | -0.000197 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.