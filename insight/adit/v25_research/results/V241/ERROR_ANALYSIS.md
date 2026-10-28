# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 33 | 0.45454545454545453 | 0.3939393939393939 |
| 0.200000 | -2 | 48 | 0.16666666666666666 | 0.20833333333333334 |
| 0.323686 | -7 | 49 | 0.12244897959183673 | 0.2653061224489796 |
| 0.400000 | -6 | 57 | 0.07017543859649122 | 0.17543859649122806 |
| 0.500000 | 3 | 59 | 0.05084745762711865 | 0.0 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | zero | 3428 | 106 | +0.003010 | +0.000065 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001310 | -0.000049 |
| orders90_exposure_band | over10 | 11489 | 3097 | +0.000369 | +0.000008 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000357 | -0.000044 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000288 | -0.000025 |
| employment_type | salaried | 33473 | 4432 | +0.000241 | -0.000030 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000170 | -0.000034 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000121 | -0.000302 |
| aux_account_months | zero | 21000 | 2461 | +0.000108 | -0.000022 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000062 | -0.000039 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000109 | -0.000030 |
| employment_type | self_employed | 20987 | 2895 | -0.000192 | -0.000001 |
| employment_type | variable_income | 15540 | 2180 | -0.000203 | -0.000077 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000258 | +0.000007 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000556 | -0.000037 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000577 | -0.000035 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000931 | -0.000122 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001172 | -0.000042 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001321 | -0.000092 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001548 | -0.000015 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002352 | +0.000099 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002379 | -0.000207 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.