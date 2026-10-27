# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 69 | 0.34782608695652173 | 0.36231884057971014 |
| 0.200000 | 16 | 120 | 0.275 | 0.14166666666666666 |
| 0.323686 | -4 | 150 | 0.11333333333333333 | 0.14 |
| 0.400000 | -5 | 204 | 0.10784313725490197 | 0.1323529411764706 |
| 0.500000 | -6 | 215 | 0.04186046511627907 | 0.06976744186046512 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.003660 | +0.001058 |
| baseline_risk_band | under005 | 29853 | 535 | +0.002756 | +0.001008 |
| employment_type | self_employed | 20987 | 2895 | +0.001120 | -0.000036 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000872 | +0.000143 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000751 | +0.000127 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000479 | +0.000037 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000459 | +0.000022 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000363 | +0.000137 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000115 | +0.000028 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000164 | +0.000058 |
| employment_type | salaried | 33473 | 4432 | -0.000182 | +0.000160 |
| aux_account_months | zero | 21000 | 2461 | -0.000320 | +0.000049 |
| employment_type | variable_income | 15540 | 2180 | -0.000349 | -0.000035 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000384 | -0.000252 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000441 | +0.000061 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000662 | +0.000078 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.000830 | -0.000039 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001145 | +0.000086 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001170 | -0.000107 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002544 | +0.000285 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002709 | -0.001011 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.002918 | -0.000312 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.