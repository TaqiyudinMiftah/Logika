# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 44 | 0.4318181818181818 | 0.3181818181818182 |
| 0.200000 | 5 | 63 | 0.31746031746031744 | 0.23809523809523808 |
| 0.323686 | -8 | 73 | 0.0684931506849315 | 0.1780821917808219 |
| 0.400000 | 0 | 91 | 0.0989010989010989 | 0.0989010989010989 |
| 0.500000 | -2 | 84 | 0.023809523809523808 | 0.047619047619047616 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001420 | -0.000285 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001261 | +0.000062 |
| employment_type | variable_income | 15540 | 2180 | +0.000941 | -0.000029 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000664 | -0.000216 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000301 | +0.000128 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000256 | -0.000060 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000100 | -0.000048 |
| employment_type | salaried | 33473 | 4432 | -0.000162 | -0.000037 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000214 | -0.000066 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000354 | -0.000041 |
| aux_account_months | zero | 21000 | 2461 | -0.000485 | -0.000036 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000509 | -0.000052 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000555 | -0.000030 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000565 | -0.000039 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000570 | -0.000016 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000623 | -0.000046 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000686 | -0.000047 |
| employment_type | self_employed | 20987 | 2895 | -0.000829 | -0.000085 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001082 | +0.000001 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001259 | -0.000317 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001565 | -0.000054 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001699 | -0.000372 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.