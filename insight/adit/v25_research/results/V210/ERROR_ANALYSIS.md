# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -7 | 239 | 0.3263598326359833 | 0.35564853556485354 |
| 0.200000 | 4 | 353 | 0.22946175637393768 | 0.21813031161473087 |
| 0.323686 | -15 | 408 | 0.10049019607843138 | 0.13725490196078433 |
| 0.400000 | -3 | 447 | 0.08501118568232663 | 0.09172259507829977 |
| 0.500000 | -4 | 424 | 0.07311320754716981 | 0.08254716981132075 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.004047 | +0.000996 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003696 | +0.000272 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.003157 | +0.000016 |
| employment_type | salaried | 33473 | 4432 | +0.002653 | +0.000099 |
| baseline_risk_band | under005 | 29853 | 535 | +0.002040 | +0.001255 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001457 | +0.000060 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.001330 | +0.000111 |
| employment_type | self_employed | 20987 | 2895 | +0.001231 | -0.000010 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000997 | +0.000074 |
| employment_type | variable_income | 15540 | 2180 | +0.000992 | +0.000340 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000894 | +0.000041 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000205 | -0.000132 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000115 | +0.000400 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000159 | +0.000105 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000426 | +0.000355 |
| aux_account_months | zero | 21000 | 2461 | -0.001014 | +0.000322 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001737 | +0.000028 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001931 | +0.000176 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002884 | -0.000031 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002980 | +0.000155 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003007 | +0.000104 |
| orders90_exposure_band | zero | 3428 | 106 | -0.006617 | -0.000040 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.