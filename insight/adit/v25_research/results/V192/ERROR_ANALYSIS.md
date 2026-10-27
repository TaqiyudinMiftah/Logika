# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -6 | 229 | 0.32751091703056767 | 0.3537117903930131 |
| 0.200000 | 1 | 330 | 0.21818181818181817 | 0.21515151515151515 |
| 0.323686 | -22 | 383 | 0.09138381201044386 | 0.14882506527415143 |
| 0.400000 | -5 | 421 | 0.08788598574821853 | 0.0997624703087886 |
| 0.500000 | 3 | 402 | 0.08208955223880597 | 0.07462686567164178 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.005020 | +0.000796 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003469 | +0.000229 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002710 | -0.000087 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.002199 | -0.000030 |
| employment_type | salaried | 33473 | 4432 | +0.001975 | +0.000060 |
| employment_type | self_employed | 20987 | 2895 | +0.001287 | -0.000123 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000546 | +0.000063 |
| employment_type | variable_income | 15540 | 2180 | +0.000413 | +0.000376 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000321 | +0.000614 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000291 | -0.000028 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000081 | -0.000016 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000050 | +0.000013 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000200 | +0.000303 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000242 | +0.000111 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001457 | -0.000094 |
| aux_account_months | zero | 21000 | 2461 | -0.001522 | +0.000322 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001523 | +0.000035 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001936 | +0.000146 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002933 | +0.000158 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003040 | +0.000104 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.003505 | -0.000028 |
| orders90_exposure_band | zero | 3428 | 106 | -0.008423 | -0.000062 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.