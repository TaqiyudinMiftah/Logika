# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 71 | 0.43661971830985913 | 0.38028169014084506 |
| 0.200000 | 7 | 114 | 0.24561403508771928 | 0.18421052631578946 |
| 0.323686 | -9 | 141 | 0.12056737588652482 | 0.18439716312056736 |
| 0.400000 | 0 | 157 | 0.10828025477707007 | 0.10828025477707007 |
| 0.500000 | -5 | 152 | 0.05263157894736842 | 0.08552631578947369 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.002717 | +0.000346 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002237 | +0.000035 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001514 | +0.000554 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000555 | -0.000342 |
| employment_type | variable_income | 15540 | 2180 | +0.000529 | +0.000079 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000479 | +0.000057 |
| employment_type | salaried | 33473 | 4432 | -0.000507 | -0.000072 |
| aux_account_months | zero | 21000 | 2461 | -0.000565 | +0.000014 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000672 | -0.000097 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000672 | -0.000061 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000790 | +0.000041 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000853 | -0.000052 |
| employment_type | self_employed | 20987 | 2895 | -0.000867 | -0.000062 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001061 | -0.000076 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001380 | -0.000057 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001846 | -0.000020 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002489 | -0.000508 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004001 | -0.000375 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.