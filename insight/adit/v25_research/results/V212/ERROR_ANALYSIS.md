# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -6 | 226 | 0.3407079646017699 | 0.3672566371681416 |
| 0.200000 | -3 | 325 | 0.2153846153846154 | 0.2246153846153846 |
| 0.323686 | -15 | 392 | 0.09438775510204081 | 0.1326530612244898 |
| 0.400000 | -2 | 439 | 0.09111617312072894 | 0.09567198177676538 |
| 0.500000 | -1 | 419 | 0.07159904534606205 | 0.07398568019093078 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.005232 | +0.000015 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.004520 | +0.000655 |
| employment_type | salaried | 33473 | 4432 | +0.002649 | +0.000016 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002554 | -0.000080 |
| employment_type | self_employed | 20987 | 2895 | +0.002447 | +0.000007 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.002366 | -0.000026 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000744 | -0.000017 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000467 | +0.000071 |
| employment_type | variable_income | 15540 | 2180 | +0.000455 | +0.000337 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000389 | +0.000266 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000184 | +0.000832 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000127 | +0.000021 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.000024 | +0.000338 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000226 | +0.000064 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000354 | -0.000001 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000943 | -0.000014 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001696 | +0.000197 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002026 | +0.000129 |
| aux_account_months | zero | 21000 | 2461 | -0.002222 | +0.000347 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.002344 | +0.000163 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002604 | -0.000001 |
| orders90_exposure_band | zero | 3428 | 106 | -0.007219 | -0.000202 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.