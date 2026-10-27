# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 3 | 0.6666666666666666 | 0.6666666666666666 |
| 0.200000 | -1 | 11 | 0.09090909090909091 | 0.18181818181818182 |
| 0.323686 | 0 | 20 | 0.2 | 0.2 |
| 0.400000 | 1 | 26 | 0.11538461538461539 | 0.07692307692307693 |
| 0.500000 | -2 | 38 | 0.02631578947368421 | 0.07894736842105263 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.000792 | +0.000017 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000767 | +0.000002 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000680 | +0.000000 |
| employment_type | self_employed | 20987 | 2895 | +0.000649 | +0.000027 |
| employment_type | variable_income | 15540 | 2180 | +0.000615 | +0.000025 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000307 | +0.000002 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000300 | +0.000000 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000184 | +0.000004 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000183 | +0.000006 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000131 | +0.000006 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000119 | +0.000004 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000021 | +0.000019 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000000 | -0.000033 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000000 | -0.000007 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000010 | +0.000037 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000027 | +0.000379 |
| employment_type | salaried | 33473 | 4432 | -0.000068 | -0.000005 |
| aux_account_months | zero | 21000 | 2461 | -0.000162 | +0.000009 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000442 | -0.000020 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000466 | +0.000018 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000624 | +0.000043 |
| orders90_exposure_band | zero | 3428 | 106 | -0.002107 | +0.000321 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.