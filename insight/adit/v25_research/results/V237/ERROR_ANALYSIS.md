# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 67 | 0.373134328358209 | 0.34328358208955223 |
| 0.200000 | 8 | 100 | 0.24 | 0.16 |
| 0.323686 | -1 | 125 | 0.104 | 0.112 |
| 0.400000 | -6 | 154 | 0.07792207792207792 | 0.11688311688311688 |
| 0.500000 | -3 | 159 | 0.0440251572327044 | 0.06289308176100629 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002866 | -0.000182 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002199 | -0.000658 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001902 | -0.000644 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001459 | +0.000218 |
| employment_type | salaried | 33473 | 4432 | +0.001351 | -0.000078 |
| employment_type | variable_income | 15540 | 2180 | +0.001205 | +0.000021 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.001191 | -0.000077 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.001092 | -0.000082 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000742 | -0.000088 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000280 | +0.000025 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000162 | -0.000096 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000091 | -0.000118 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000507 | +0.000128 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000620 | -0.000093 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000837 | -0.000061 |
| employment_type | self_employed | 20987 | 2895 | -0.000870 | -0.000067 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001155 | +0.000013 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001220 | +0.000057 |
| orders90_exposure_band | zero | 3428 | 106 | -0.001403 | +0.000000 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001520 | +0.000107 |
| aux_account_months | zero | 21000 | 2461 | -0.001532 | +0.000133 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001569 | -0.000222 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.