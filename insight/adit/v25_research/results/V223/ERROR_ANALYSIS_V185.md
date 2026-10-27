# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 12 | 252 | 0.3888888888888889 | 0.3412698412698413 |
| 0.200000 | -1 | 372 | 0.21774193548387097 | 0.22043010752688172 |
| 0.323686 | 10 | 441 | 0.1360544217687075 | 0.11337868480725624 |
| 0.400000 | 2 | 474 | 0.08227848101265822 | 0.07805907172995781 |
| 0.500000 | 5 | 472 | 0.0826271186440678 | 0.07203389830508475 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003162 | -0.000106 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002574 | -0.000064 |
| aux_account_months | zero | 21000 | 2461 | +0.001938 | -0.000242 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001508 | -0.000061 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000984 | +0.000067 |
| employment_type | variable_income | 15540 | 2180 | +0.000342 | -0.000344 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000147 | -0.000022 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000326 | -0.000088 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000420 | -0.000114 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000442 | -0.000007 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000904 | +0.000890 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001411 | +0.000097 |
| employment_type | salaried | 33473 | 4432 | -0.001769 | -0.000046 |
| baseline_risk_band | under005 | 29965 | 546 | -0.002008 | -0.000379 |
| employment_type | self_employed | 20987 | 2895 | -0.002075 | +0.000073 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002295 | +0.000131 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.003024 | -0.001082 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003943 | -0.000417 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.